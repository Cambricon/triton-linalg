//===- InferAxisInfoInterface.cpp - Infer axis info -------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface for axis info analysis.
//
//===----------------------------------------------------------------------===//

#include "triton-linalg/Interfaces/InferAxisInfoInterface.h"

#include <numeric>
#include <string>
#include <utility>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "triton-linalg/Interfaces/InferAxisInfoInterface.cpp.inc"

using namespace mlir;
using namespace mlir::triton;

template <typename T> static int64_t leastCommonMultiple(T a, T b) {
  return a * (b / std::gcd<T, T>(a, b));
}

//===--------------------------------------------------------------------===//
// The main logical is modified from
// lib/Analysis/AxisInfo.cpp in the triton repo.
//===--------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

AxisInfo AxisInfo::overrideByHint(Operation *op) const {
  return overrideAxisInfoByHint(op, divisibility, stride, strideValue,
                                constantValue);
}

AxisInfo AxisInfo::getPessimisticValueState(Value value) {
  auto rank = 1;
  if (TensorType ty = value.getType().dyn_cast<TensorType>())
    rank = ty.getRank();

  AxisInfo ret(DimVectorT(rank, kInitValue), DimVectorT(rank, kInitValue),
               DimVectorT(rank, kStrideValueInitValue));
  BlockArgument blockArg = value.dyn_cast<BlockArgument>();
  if (!blockArg || !blockArg.getOwner()->isEntryBlock()) {
    return ret;
  }

  Operation *op = blockArg.getOwner()->getParentOp();
  FunctionOpInterface func = dyn_cast<FunctionOpInterface>(op);
  if (!func) {
    return ret;
  }

  // Last of attributes that we care about.
  AxisInfo::DimVectorT contiguity, divisibility, constancy;
  SmallVector<std::pair<AxisInfo::DimVectorT *, std::string>> retVecs;
  retVecs.push_back({&contiguity, "tt.contiguity"});
  retVecs.push_back({&divisibility, "tt.divisibility"});
  retVecs.push_back({&constancy, "tt.constancy"});
  // Initialize attributes one by one.
  for (auto [vec, attrName] : retVecs) {
    Attribute attr = func.getArgAttr(blockArg.getArgNumber(), attrName);
    if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>())
      *vec = AxisInfo::DimVectorT(rank, intAttr.getValue().getZExtValue());
    if (auto denseAttr = attr.dyn_cast_or_null<DenseElementsAttr>()) {
      auto vals = denseAttr.getValues<int>();
      *vec = AxisInfo::DimVectorT(vals.begin(), vals.end());
    }
  }

  if (divisibility.empty()) {
    divisibility = AxisInfo::DimVectorT(rank, kInitValue);
  }

  if (!constancy.empty()) {
    assert(contiguity.empty() &&
           "Get tt.constancy and tt.contiguity attribute at the same arg");
    return AxisInfo(divisibility, constancy, DimVectorT(rank, 0));
  }
  if (!contiguity.empty()) {
    return AxisInfo(divisibility, contiguity, DimVectorT(rank, 1));
  }
  return AxisInfo(divisibility, DimVectorT(rank, kInitValue),
                  DimVectorT(rank, kStrideValueInitValue));
}

AxisInfo AxisInfo::join(const AxisInfo &lhs, const AxisInfo &rhs) {
  auto lhsRank = lhs.getRank();
  auto rhsRank = rhs.getRank();
  // When rank equals to zero, means in unintialized state, just return the
  // other.
  if (lhsRank == 0)
    return rhs;
  if (rhsRank == 0)
    return lhs;

  assert(lhsRank == rhsRank);
  DimVectorT divisibility(lhsRank, kInitValue);
  DimVectorT stride(lhsRank, kInitValue);
  DimVectorT strideValue(lhsRank, kStrideValueInitValue);
  for (auto d = 0; d < lhsRank; ++d) {
    divisibility[d] =
        leastCommonMultiple(lhs.getDivisibility(d), rhs.getDivisibility(d));
    stride[d] = leastCommonMultiple(lhs.getStride(d), rhs.getStride(d));
    if (lhs.strideValue[d] != kStrideValueInitValue &&
        rhs.strideValue[d] != kStrideValueInitValue &&
        lhs.strideValue[d] == rhs.strideValue[d]) {
      strideValue[d] = lhs.strideValue[d];
    }
  }
  std::optional<int64_t> constantValue;
  if (lhs.getConstantValue().has_value() &&
      rhs.getConstantValue().has_value() &&
      lhs.getConstantValue() == rhs.getConstantValue())
    constantValue = lhs.getConstantValue();
  return AxisInfo(divisibility, stride, strideValue, constantValue);
}

void AxisInfo::print(raw_ostream &os) const {
  auto print = [&](StringRef name, const AxisInfo::DimVectorT &vec) {
    os << name << " = [";
    llvm::interleaveComma(vec, os);
    os << "]";
  };
  print("divisibility", divisibility);
  print(", stride", stride);
  print(", stride_value", strideValue);
  os << ", constant_value = ";
  if (constantValue)
    os << *constantValue;
  else
    os << "<none>";
}

AxisInfo
triton::overrideAxisInfoByHint(Operation *op,
                                const AxisInfo::DimVectorT &knownDivisibility,
                                const AxisInfo::DimVectorT &knownStride,
                                const AxisInfo::DimVectorT &knownStrideValue,
                                std::optional<int64_t> constantValue) {
  AxisInfo::DimVectorT divisibility = knownDivisibility, stride = knownStride,
                       strideValue = knownStrideValue;
  if (Attribute attr = op->getAttr("tt.divisibility")) {
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    divisibility = AxisInfo::DimVectorT(vals.begin(), vals.end());
  }
  if (Attribute attr = op->getAttr("tt.contiguity")) {
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    stride = AxisInfo::DimVectorT(vals.begin(), vals.end());
    strideValue = AxisInfo::DimVectorT(vals.size(), 1);
  }
  if (Attribute attr = op->getAttr("tt.constancy")) {
    assert(!op->getAttr("tt.contiguity") &&
           "Get tt.constancy and tt.contiguity attribute at the same op");
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    stride = AxisInfo::DimVectorT(vals.begin(), vals.end());
    strideValue = AxisInfo::DimVectorT(vals.size(), 0);
  }
  return AxisInfo(divisibility, stride, strideValue, constantValue);
}
