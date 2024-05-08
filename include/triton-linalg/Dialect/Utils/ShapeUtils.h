//===- ShapeUtils.h - Shape compute utils. ----------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_UTILS_SHAPEUTILS_H
#define TRITON_LINALG_DIALECT_UTILS_SHAPEUTILS_H
#include <stdint.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
namespace triton {
/// Return input is continuously increasing sequence or not.
/// Example:
///   [1, 2, 3] -> true
///   [3, 2, 1] -> false
///   [1, 3] -> false
bool isConsecutive(llvm::ArrayRef<int64_t> value);

/// Get dim as value or attr from given value's dim dimension.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Get dim as value from given value's dim dimension.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Get dims as value or attr from given value's dim dimension.
SmallVector<OpFoldResult> getDims(OpBuilder &builder, Location loc,
                                  Value shapedTypeValue);

/// Get dims as value or attr from given value's dim dimension.
SmallVector<Value> getDimsValue(OpBuilder &builder, Location loc,
                                Value shapedTypeValue);

/// Get dynamic dims as value from given tensor or memref value.
SmallVector<Value> getDynamicDimsValue(OpBuilder &builder, Location loc,
                                       Value val);

/// An OpBuilder version of materialize in linalg.
Value materializeOpFoldResult(OpBuilder &builder, Location loc,
                              OpFoldResult opFoldResult);

/// Return whether value is a scalar.
bool isScalar(Value val);

/// Add a unit dimension to the last dim.
///
/// Example 1:
/// ```mlir
///   %0 = foo ... : tensor<d0xd1x..dnxf32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<d0xd1x..dnxf32>
///   %2 = tensor.expand_shape %1 [[0] [1] ... [n, n+1]] :
///        tensor<d0xd1x...dnxf32> to tensor<d0xd1x...xdnx1xf32>
/// ```
///
/// Example 2:
/// ```mlir
///   %0 = foo ... : tensor<f32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<f32>
///   %2 = tensor.expand_shape %1 [[]] : tensor<f32> to tensor<1xf32>
/// ```
Value appendUnitDim(OpBuilder &b, Location loc, Value value);

/// Add a unit dimension to the front.
///
/// Example 1:
/// ```mlir
///   %0 = foo ... : tensor<d0xd1x..dnxf32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<d0xd1x..dnxf32>
///   %1 = tensor.expand_shape %0 [[0, 1] ... [n] [n+1]] :
///        tensor<d0xd1x...dnxf32> to tensor<1xd0xd1x...xdnxf32>
/// ```
///
/// Example 2:
/// ```mlir
///   %0 = foo ... : tensor<f32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<f32>
///   %1 = tensor.expand_shape %0 [[]] : tensor<f32> to tensor<1xf32>
/// ```
Value prependUnitDim(OpBuilder &b, Location loc, Value value);

/// Drop the first dimension. It is the precondition for this function that the
/// size of first dimension is 1.
///
/// Example 1:
/// ```mlir
///   %0 = foo ... : tensor<1xd1x..dnxf32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<1xd1xd1x..dnxf32>
///   %1 = tensor.collapse_shape %0 [[0, 1] ... [n] [n+1]] :
///        tensor<1xd1x...dnxf32> to tensor<d1x...xdnxf32>
/// ```
///
/// Example 2:
/// ```mlir
///   %0 = foo ... : tensor<f32>
/// ```
/// is reshaped to:
/// ```mlir
///   %0 = foo ... : tensor<1xf32>
///   %1 = tensor.expand_shape %0 [] : tensor<1xf32> to tensor<f32>
/// ```
Value dropUnitFirstDim(OpBuilder &b, Location loc, Value value);

/// Collapse the last n dims to one dim.
///
/// Example 1 (n = 0):
/// ```mlir
/// %0 = foo ... : tensor<f32>
/// ```
/// is collapsed to:
/// ```mlir
/// %0 = foo ... : tensor<f32>
/// %1 = tensor.expand_shape %0 [] : tensor<f32> into tensor<1xf32>
/// ```
///
/// Example 2 (n = 1):
/// ```mlir
/// %0 = foo ... : tensor<s0xs1x..xsnxf32>
/// ```
/// is collapsed to:
/// ```mlir
/// %0 = foo ... : tensor<s0xs1x..xsnxf32>
/// ```
///
/// Example 3 (n = 2):
/// ```mlir
/// %0 = foo ... : tensor<s0xs1x..xsn-1xsnxf32>
/// ```
/// is collapsed to (sn-1' = sn-1xsn):
/// ```mlir
/// %0 = foo ... : tensor<s0xs1x..xsn-1xsnxf32>
/// %1 = tensor.collapse_shape %n [[0] [1] ... [sn-1, sn]
///      : tensor<s0xs1x..xsn-1xsnxf32> to tensor<s0xs1x..xsn-1'xf32>
/// ```
Value collapseLastNDimsToOneDim(OpBuilder &b, Location loc, Value value,
                                int64_t n);

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_UTILS_SHAPEUTILS_H
