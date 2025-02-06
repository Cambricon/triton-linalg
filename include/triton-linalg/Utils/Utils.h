//===- Utils.h - Transform utils. -------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_UTILS_UTILS_H
#define TRITON_LINALG_UTILS_UTILS_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class RewriterBase;
class OpBuilder;
namespace triton {
/// Create reassociation maps for expandedShape and collapsedShape.
/// Example 0: expandedShape[1, 16, 1] collapsedShape[16] =>
///   reassociationMap{{d0,d1,d2}}
/// Example 1: expandedShape[1, 16, 32] collapsedShape[16, 32] =>
///   reassociationMap{{d0,d1},{d2}}
bool createReassociationMaps(
    OpBuilder &builder, llvm::ArrayRef<int64_t> expandedShape,
    llvm::ArrayRef<int64_t> collapsedShape,
    llvm::SmallVector<ReassociationExprs, 4> &reassociationMap);

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

/// Produce result = div(lhs, rhs). If both OFRs are Integer Attributes, result
/// is an Integer Attribute. Otherwise, insert the arith.divsi instruction if
/// needed and use its result value.
OpFoldResult divOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
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

#endif // TRITON_LINALG_UTILS_UTILS_H
