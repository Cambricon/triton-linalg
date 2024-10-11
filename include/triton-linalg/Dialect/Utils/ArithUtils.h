//===- ArithUtils.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_UTILS_ARITHUTILS_H
#define TRITON_LINALG_DIALECT_UTILS_ARITHUTILS_H
#include <stdint.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"

namespace llvm {
class APInt;
} // namespace llvm

namespace mlir {
class OpBuilder;
namespace triton {
/// Create a constant of type `type` at location `loc` whose value is `value`
/// (an APInt or APFloat whose type must match the element type of `type`).
/// If `type` is a shaped type, create a splat constant of the given value.
/// Constants are folded if possible.
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  const APInt &value);
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  int64_t value);
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  const APFloat &value);

/// Get splat value from the arith.constant op.
FailureOr<Value> getSplatValue(OpBuilder &builder, arith::ConstantOp op);
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_UTILS_ARITHUTILS_H
