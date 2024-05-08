//===- InferAxisInfoInterface.h - Infer axis info ---------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface for axis info analysis.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_H
#define TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_H

#include <assert.h>
#include <optional>
#include <stddef.h>
#include <stdint.h>

#include "mlir/IR/OpDefinition.h" // IWYU pragma: keep
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class raw_ostream;
} // namespace llvm
namespace mlir {
class MLIRContext;
class Operation;
} // namespace mlir

namespace mlir {
namespace triton {

//===--------------------------------------------------------------------===//
// The main logical is modified from
// include/triton/Analysis/AxisInfo.h in the triton repo.
//===--------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfo {
public:
  using DimVectorT = SmallVector<int64_t, 4>;
  constexpr static int64_t kInitValue = 1;
  constexpr static int64_t kStrideValueInitValue = -1;

public:
  AxisInfo() : AxisInfo({}, {}, {}) {}
  AxisInfo(ArrayRef<int64_t> knownDivisibility, ArrayRef<int64_t> knownStride,
           ArrayRef<int64_t> knownStrideValue)
      : AxisInfo(knownDivisibility, knownStride, knownStrideValue, {}) {}
  AxisInfo(ArrayRef<int64_t> knownDivisibility, ArrayRef<int64_t> knownStride,
           ArrayRef<int64_t> knownStrideValue,
           std::optional<int64_t> knownConstantValue) {
    divisibility.append(knownDivisibility.begin(), knownDivisibility.end());
    stride.append(knownStride.begin(), knownStride.end());
    strideValue.append(knownStrideValue.begin(), knownStrideValue.end());
    constantValue = knownConstantValue;
    rank = stride.size();
    assert(knownDivisibility.size() == static_cast<size_t>(rank));
    assert(knownStride.size() == static_cast<size_t>(rank));
    assert(knownStrideValue.size() == static_cast<size_t>(rank));
  }

  /// Accessors.
  int64_t getContiguity(size_t dim) const {
    if (strideValue[dim] == 1)
      return stride[dim];
    return 1;
  }
  int64_t getConstancy(size_t dim) const {
    if (strideValue[dim] == 0)
      return stride[dim];
    return 1;
  }

  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int64_t getStride(size_t dim) const { return stride[dim]; }
  const DimVectorT &getStride() const { return stride; }

  int64_t getStrideValue(size_t dim) const { return strideValue[dim]; }
  const DimVectorT &getStrideValue() const { return strideValue; }

  int getRank() const { return rank; }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  bool isContiguousDim(ArrayRef<int64_t> shape, int dim) const {
    return getContiguity(dim) == shape[dim];
  }

  bool isStrideDim(ArrayRef<int64_t> shape, int dim) const {
    return getStride(dim) == shape[dim];
  }

  bool isNonContiguousNonConstantStrideDim(ArrayRef<int64_t> shape,
                                           int dim) const {
    return getStride(dim) == shape[dim] && !isContiguousDim(shape, dim) &&
           !isConstantDim(shape, dim);
  }

  bool isConstantDim(ArrayRef<int64_t> shape, int dim) const {
    return getConstancy(dim) == shape[dim];
  }

  /// Comparison.
  bool operator==(const AxisInfo &other) const {
    return (divisibility == other.divisibility) && (stride == other.stride) &&
           (strideValue == other.strideValue) &&
           (constantValue == other.constantValue) && (rank == other.rank);
  }

  AxisInfo overrideByHint(Operation *op) const;

  /// The pessimistic value state of the contiguity is unknown.
  static AxisInfo getPessimisticValueState(MLIRContext *context = nullptr) {
    return AxisInfo();
  }
  static AxisInfo getPessimisticValueState(Value value);

  /// The gcd of both arguments for each dimension.
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

  void print(raw_ostream &os) const;

private:
  /// The _divisibility_ information maps the `d`-th
  /// dimension to the largest power-of-two that
  /// divides the first element of all the values along it.
  /// For example:
  /// [10, 11, 12, 13, 18, 19, 20, 21]
  /// [20, 21, 22, 23, 28, 29, 30, 31]
  /// would have divisibility [1, 2]
  /// and
  /// [12, 16, 20, 24]
  /// [13, 17, 21, 25]
  /// [14, 18, 22, 26]
  /// [15, 19, 23, 27]
  /// would have divisibility [4, 1].
  DimVectorT divisibility;

  /// The _stride_ information maps the `d`-th
  /// dimension to the length of the shortest
  /// sequence of integers of the same stride along it.
  /// Suppose we have an array of N elements,
  /// with a stride value C,
  /// the array can be divided into a list of
  /// N/C sequences of C subsequence of integers of the same stride.
  /// For example:
  /// [10, 11, 12, 13, 18, 19, 20, 21]
  /// [20, 21, 22, 23, 28, 29, 30, 31]
  /// Would have stride [2, 4].
  /// and
  /// [12, 16, 20, 24]
  /// [13, 17, 21, 25]
  /// [14, 18, 22, 26]
  /// [15, 19, 23, 27]
  /// [18, 22, 26, 30]
  /// [19, 23, 27, 31]
  /// Would have stride [2, 4].
  DimVectorT stride;

  /// The _stride_value_ information maps the `d`-th
  /// dimension to the stride value of the shortest
  /// sequence of integers of the same stride along it.
  /// Suppose we have an array of N elements,
  /// with a stride value C,
  /// the array can be divided into a list of
  /// N/C sequences of C subsequence of integers of the same stride.
  /// For example:
  /// [10, 11, 12, 13, 18, 19, 20, 21]
  /// [20, 21, 22, 23, 28, 29, 30, 31]
  /// Would have strideValue [10, 1].
  /// and
  /// [12, 16, 20, 24]
  /// [13, 17, 21, 25]
  /// [14, 18, 22, 26]
  /// [15, 19, 23, 27]
  /// [18, 22, 26, 30]
  /// [19, 23, 27, 31]
  /// Would have strideValue [1, 4].
  DimVectorT strideValue;

  /// The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;

  // Number of dimensions of the lattice.
  int rank{};
};

/// The type of the `setResultAxisInfo` callback provided to ops implementing
/// InferAxisInfoInterface. It should be called once for each op result
/// value and be passed the AxisInfo corresponding to that value.
using SetAxisInfoFn = function_ref<void(Value, const AxisInfo &)>;

inline raw_ostream &operator<<(raw_ostream &os, const AxisInfo &info) {
  info.print(os);
  return os;
}

/// Override axis info by use provided attributes hint.
AxisInfo overrideAxisInfoByHint(Operation *op,
                                const AxisInfo::DimVectorT &knownDivisibility,
                                const AxisInfo::DimVectorT &knownStride,
                                const AxisInfo::DimVectorT &knownStrideValue,
                                std::optional<int64_t> constantValue);

} // namespace triton
} // namespace mlir

/// Include the generated interface declarations.
#include "triton-linalg/Interfaces/InferAxisInfoInterface.h.inc" // IWYU pragma: export

#endif // TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_H
