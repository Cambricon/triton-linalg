//===- OpFoldResultUtils.h - Some utilities for OpFoldResult ----*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_UTILS_OPFOLDRESULTUTILS_H
#define TRITON_LINALG_DIALECT_TRITON_UTILS_OPFOLDRESULTUTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class OpBuilder;
namespace triton {

Value castToIndexType(OpBuilder &b, Location loc, OpFoldResult ofr);
SmallVector<Value> castToIndexType(OpBuilder &b, Location loc,
                                   ArrayRef<OpFoldResult> ofrs);

/// Process addition of two OFRs. If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.addi instruction if
/// needed and use its result value.
OpFoldResult addOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                     OpBuilder &rewriter);

/// Produce result = lhs - rhs. If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.subi instruction if
/// needed and use its result value.
OpFoldResult subOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                     OpBuilder &rewriter);

/// Produce result = mul(lhs, rhs). If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.mulsi instruction if
/// needed and use its result value.
OpFoldResult mulOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                     OpBuilder &rewriter);

/// Produce result = min(lhs, rhs). If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.minsi instruction if
/// needed and use its result value.
OpFoldResult minOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                     OpBuilder &rewriter);

/// Produce result = max(lhs, rhs). If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.maxsi instruction if
/// needed and use its result value.
OpFoldResult maxOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                     OpBuilder &rewriter);
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_UTILS_OPFOLDRESULTUTILS_H
