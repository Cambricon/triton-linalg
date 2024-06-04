//===- PointerMetaInfoTracker.h - Trace the pointer pattern ----*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERMETAINFOTRACKER_H
#define TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERMETAINFOTRACKER_H

#include <stddef.h>
#include <stdint.h>

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class ConversionPatternRewriter;
namespace triton {

/// Data structure used to extract the tensor pointer used for load and store.
/// base/sizes/offsets/strides/order are the meta info extracted from
/// MakeTensorPtrOp. The initial value of offsets is extracted from MakeTensorOp
/// and computed through AdvanceOp.
class TensorPointerMetaInfoTracker {
public:
  size_t getRank() const { return sizes.size(); }
  Value getBase() const { return base; }
  ArrayRef<OpFoldResult> getSizes() const { return sizes; }
  ArrayRef<OpFoldResult> getStrides() const { return strides; }
  ArrayRef<OpFoldResult> getOffsets() const { return offsets; }
  ArrayRef<int32_t> getOrder() const { return order; }

  LogicalResult parse(Value operand, Location loc,
                      ConversionPatternRewriter &rewriter);

private:
  template <typename OpTy>
  LogicalResult parseOp(OpTy op, Location loc,
                        ConversionPatternRewriter &rewriter);

  Value base;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> offsets;
  ArrayRef<int32_t> order;
};

/// Data structure used to extract the normal pointer used for load and store.
/// base/offset are the meta info extracted from the operator chain of
/// the pointer calculation.
/// This analysis assume that RewriteTritonTensorPointer pass has already been
/// used to canonicalize the pointer calculate form, so we just need to trace
/// tt.addptr/tt.bitcast/tt.splat.
class PointerMetaInfoTracker {
public:
  Value getBase() const { return base; }
  Value getOffset() const { return offset; }

  FailureOr<bool> parse(Value operand, Location loc,
                        ConversionPatternRewriter &rewriter);
private:
  template <typename OpTy>
  LogicalResult parseOp(OpTy op, Location loc,
                        ConversionPatternRewriter &rewriter);

  Value base;
  Value offset;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERMETAINFOTRACKER_H
