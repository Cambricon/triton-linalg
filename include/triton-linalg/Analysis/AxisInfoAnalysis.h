//===- AxisInfoAnalysis.h - Axis info Analysis ------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for axis info inference.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_ANALYSIS_AXISINFOANALYSIS_H
#define TRITON_LINALG_DIALECT_TRITON_ANALYSIS_AXISINFOANALYSIS_H

#include "triton-linalg/Interfaces/InferAxisInfoInterface.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include <type_traits>

namespace mlir {
class Operation;
class DataFlowSolver;
} // namespace mlir

namespace mlir {
namespace triton {

class AxisInfoLattice : public mlir::dataflow::Lattice<AxisInfoExt> {
public:
  using Lattice::Lattice;
};

//===--------------------------------------------------------------------===//
// The main logical is modified from
// include/triton/Analysis/AxisInfo.h in the triton repo.
//===--------------------------------------------------------------------===//
class AxisInfoAnalysisExt
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<AxisInfoLattice> {
public:
  AxisInfoAnalysisExt(mlir::DataFlowSolver &solver);
  using mlir::dataflow::SparseForwardDataFlowAnalysis<
      AxisInfoLattice>::getLatticeElement;

  void visitOperation(Operation *op, ArrayRef<const AxisInfoLattice *> operands,
                      ArrayRef<AxisInfoLattice *> results) override;

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<AxisInfoLattice *> argLattices,
                                    unsigned firstIndex) override;

  void setToEntryState(AxisInfoLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(AxisInfoExt::getPessimisticValueState(
                           lattice->getPoint())));
  }
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_ANALYSIS_AXISINFOANALYSIS_H
