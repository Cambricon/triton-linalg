//===- AxisInfoAnalysis.cpp - Axis info Analysis ----------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for axis info inference.
//
//===----------------------------------------------------------------------===//

#include "triton-linalg/Analysis/AxisInfoAnalysis.h"
#include "triton-linalg/Dialect/Triton/Interfaces/InferAxisInfoInterface.h"

#include <assert.h>
#include <numeric>
#include <optional>
#include <stdint.h>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

namespace mlir {
class DataFlowSolver;
class RegionSuccessor;
enum class ChangeResult;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

#define DEBUG_TYPE "axis-info-analysis"

//===--------------------------------------------------------------------===//
// The main logical is modified from
// lib/Analysis/AxisInfo.cpp in the triton repo.
//===--------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AxisInfoLattice
//===----------------------------------------------------------------------===//

ChangeResult AxisInfoLattice::join(const AxisInfoExt &rhs) {
  if (!initialized) {
    initialized = true;
    auto &kepval = getValue();
    if (kepval == rhs)
      return ChangeResult::NoChange;
    kepval = rhs;
    return ChangeResult::Change;
  }
  return mlir::dataflow::Lattice<AxisInfoExt>::join(rhs);
}

//===----------------------------------------------------------------------===//
// AxisInfoAnalysisExt
//===----------------------------------------------------------------------===//

AxisInfoAnalysisExt::AxisInfoAnalysisExt(mlir::DataFlowSolver &solver)
    : dataflow::SparseForwardDataFlowAnalysis<AxisInfoLattice>(solver) {}

void AxisInfoAnalysisExt::visitOperation(
    Operation *op, ArrayRef<const AxisInfoLattice *> operands,
    ArrayRef<AxisInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring axis info for " << *op << "\n");

  auto inferrable = dyn_cast<InferAxisInfoInterface>(op);
  if (!inferrable) {
    return setAllToEntryStates(results);
  }

  auto joinCallback = [op, results, this](Value v, const AxisInfoExt &info) {
    auto result = v.dyn_cast<OpResult>();
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));
    LLVM_DEBUG(llvm::dbgs() << "Inferred axis info " << info << "\n");

    AxisInfoLattice *lattice = results[result.getResultNumber()];
    ChangeResult changed = lattice->join(info.overrideByHint(op));
    if (lattice->getValue().getRank() == 0) {
      setToEntryState(lattice);
    }
    propagateIfChanged(lattice, changed);
  };

  SmallVector<AxisInfoExt, 4> argInfos(
      llvm::map_range(operands, [this](const AxisInfoLattice *val) {
        // As DataFlowFramework will not pass the lattice from the operand of
        // scf.yield to the result of scf.for, then get an empty AxisInfo,
        // set it to entry state to avoid special logic in func inferAxisInfos.
        if (val->getValue().getRank() == 0)
          setToEntryState((AxisInfoLattice *)val);
        return val->getValue();
      }));
  inferrable.inferAxisInfos(argInfos, joinCallback);
}

void AxisInfoAnalysisExt::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<AxisInfoLattice *> argLattices, unsigned firstIndex) {

  auto getRank = [](Type type) {
    auto rank = 1;
    if (TensorType ty = type.dyn_cast<TensorType>())
      rank = ty.getRank();
    return rank;
  };

  auto joinCallback = [&getRank, this](Value val, int64_t divHint) {
    auto rank = getRank(val.getType());
    auto *lattice = getLatticeElement(val);
    propagateIfChanged(
        lattice, lattice->join(AxisInfoExt(
                     AxisInfoExt::DimVectorT(rank, divHint),
                     AxisInfoExt::DimVectorT(rank, AxisInfoExt::kInitValue),
                     AxisInfoExt::DimVectorT(
                         rank, AxisInfoExt::kStrideValueInitValue))));
  };

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    std::optional<Value> iv = forOp.getSingleInductionVar();
    auto lowerBound = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto step = forOp.getStep().getDefiningOp<arith::ConstantOp>();
    if (!iv || !lowerBound || !step) {
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }

    auto lowerBoundVal =
        lowerBound.getValue().cast<IntegerAttr>().getValue().getZExtValue();
    auto stepVal =
        step.getValue().cast<IntegerAttr>().getValue().getZExtValue();
    auto divHint = AxisInfoExt::kInitValue;
    auto k = std::gcd(lowerBoundVal, stepVal);
    if (k != 0)
      divHint = k;
    return joinCallback(iv.value(), divHint);
  }
  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}
