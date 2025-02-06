//===- MaskTracker.h - Trace the mask pattern ------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_UTILS_MASKTRACKER_H
#define TRITON_LINALG_DIALECT_TRITON_UTILS_MASKTRACKER_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <stdint.h>

namespace mlir {
namespace triton {
class MaskTracker;

raw_ostream &operator<<(raw_ostream &os, const MaskTracker &tracker);
//===--------------------------------------------------------------------===//
// The main code is modified from triton-to-linalg branch in triton repo.
//===--------------------------------------------------------------------===//

// Data structure used to decode the pattern in a mask used for load and store.
// start and end field represent the start and end index of a range (produced
// by make_range, addi, etc.). While multi-dimensional data is possible, we
// assume range comparison can only be done on 1 dimension at a time (and
// results of range comparions across dimensions can be combined), hence start
// and end are not vectors. dims represents the real access size for ld/st
// (instead of the tensor/memref size specified by the IR). scalar is a shortcut
// used when the entire info contains a single scalar value.
//
// The general lifetime of this data structure is roughly:
// 1. A range is created by make_range and optionally operated on by addi w/
// result of splat, expand_dims, etc. During this phase, either (1) both start
// and end are populated, or (2) scalar is populated. Only one of the dimensions
// (that contains the range) can have dim > 1.
// 2. Result from step 1 is compared with a another MaskTracker that represents
// a scalar value. The resulting info only has dims populated.
// 3. Optionally, result from step 2 can be broadcasted and anded with other
// results from step 2. The resulting info only has dims populated.
//
// Example of creating 2D mask:
//  mask = (rows[:, None] < M) & (cols[None, :] < N)
class MaskTracker {
public:
  // When sizes is not empty and has sizes if nullptr, return true.
  // When sizes is empty return false, because
  // we think scalar value tracker is failed.
  bool hasFailedDim() const;
  int64_t getRank() const { return dims.size(); }
  ArrayRef<OpFoldResult> getSizes() const { return dims; }
  ArrayRef<OpFoldResult> getStarts() const { return starts; }
  void dump() const { llvm::errs() << *this << "\n"; }

  /// Recursively parse a Value; call the corresponding function based on the
  /// defining operation and Value type.
  void parse(Value operand, Location loc, RewriterBase &rewriter);
  void setDimensionStatus(int64_t dim, OpFoldResult start, OpFoldResult end,
                          OpFoldResult dimSize);
  friend raw_ostream &operator<<(raw_ostream &os, const MaskTracker &tracker);

private:
  SmallVector<OpFoldResult> starts, ends, dims;
};

inline raw_ostream &operator<<(raw_ostream &os, const MaskTracker &tracker) {
  os << "tracker result: {\n";
  for (auto idx : llvm::seq<int64_t>(0, tracker.dims.size())) {
    os << "axis: " << idx << ", dims: " << tracker.dims[idx]
       << ", start: " << tracker.starts[idx] << ", end: " << tracker.ends[idx]
       << "\n";
  }
  return os << "}";
}

/// Data structure used to decode range cmp pattern to get start, end and axis.
/// start and end field represent the start and end index of a range (produced
/// by make_range, addi, etc.).
///
/// Example of creating 2D range:
///  range = rows[:, None]
class RangeTracker {
public:
  /// Recursively parse a Value; call the corresponding function based on the
  /// defining operation and Value type.
  LogicalResult parse(Value operand, Location loc, RewriterBase &rewriter);

  int64_t getAxis() const { return axis; }
  OpFoldResult getStart() const { return start; }
  OpFoldResult getEnd() const { return end; }

private:
  OpFoldResult start, end;
  int64_t axis = -1;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_UTILS_MASKTRACKER_H
