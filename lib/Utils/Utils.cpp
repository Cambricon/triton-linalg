//===- OpFoldResultUtils.cpp - Some utilities for OpFoldResult --*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "triton-linalg/Utils/Utils.h"

#include <algorithm>
#include <assert.h>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::triton;

bool mlir::triton::createReassociationMaps(
    OpBuilder &builder, llvm::ArrayRef<int64_t> expandedShape,
    llvm::ArrayRef<int64_t> collapsedShape,
    llvm::SmallVector<ReassociationExprs, 4> &reassociationMap) {
  if (collapsedShape.empty()) {
    reassociationMap = {};
    return true;
  }

  // As tensor.expand_shape/tensor.collapse_shape expected rank
  // expansion/reduction.
  if (expandedShape.size() == collapsedShape.size())
    return false;
  if (ShapedType::isDynamicShape(expandedShape) ||
      ShapedType::isDynamicShape(collapsedShape))
    return false;

  reassociationMap.resize(collapsedShape.size());
  unsigned currExpandDim = 0, currCollapseDim = 0;
  while (currExpandDim < expandedShape.size() &&
         currCollapseDim < collapsedShape.size()) {
    int64_t dstSize = collapsedShape[currCollapseDim];
    int64_t srcSize = expandedShape[currExpandDim];
    while (srcSize < dstSize && currExpandDim < expandedShape.size()) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      srcSize *= expandedShape[currExpandDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currCollapseDim == collapsedShape.size() - 1 ||
          collapsedShape[currCollapseDim + 1] != 1) {
        while (currExpandDim < expandedShape.size() &&
               expandedShape[currExpandDim] == 1) {
          reassociationMap[currCollapseDim].push_back(
              builder.getAffineDimExpr(currExpandDim++));
        }
      }
    }
    currCollapseDim++;
  }
  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currExpandDim == expandedShape.size() &&
         currCollapseDim == collapsedShape.size();
}

Value triton::castToIndexType(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto value = ofr.dyn_cast<Value>()) {
    if (!value.getType().isa<IndexType>())
      return b.createOrFold<arith::IndexCastOp>(loc, b.getIndexType(), value);
    return value;
  }
  auto attr = ofr.dyn_cast<Attribute>().dyn_cast<IntegerAttr>();
  assert(attr && "expect the op fold result casts to an integer attribute");
  return b.create<arith::ConstantIndexOp>(loc, attr.getValue().getSExtValue())
      .getResult();
}

SmallVector<Value> triton::castToIndexType(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> ret;
  for (auto ofr : ofrs) {
    ret.push_back(castToIndexType(b, loc, ofr));
  }
  return ret;
}

OpFoldResult triton::addOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                             OpBuilder &builder) {
  auto lhsIntAttr = getConstantIntValue(lhs);
  auto rhsIntAttr = getConstantIntValue(rhs);

  // Shortcut for special cases.
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // Both lhs and rhs are constants, return result directly.
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  // Otherwise, need to create instructions to calculate new attribute value.
  auto lhsValue = castToIndexType(builder, loc, lhs);
  auto rhsValue = castToIndexType(builder, loc, rhs);
  return builder.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult triton::subOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                             OpBuilder &builder) {
  auto lhsIntAttr = getConstantIntValue(lhs);
  auto rhsIntAttr = getConstantIntValue(rhs);

  // Shortcut for special cases.
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // Both lhs and rhs are constants, return result directly.
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // Otherwise, need to create instructions to calculate new attribute value.
  auto lhsValue = castToIndexType(builder, loc, lhs);
  auto rhsValue = castToIndexType(builder, loc, rhs);
  return builder.create<arith::SubIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult triton::minOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                             OpBuilder &builder) {
  auto lhsIntAttr = getConstantIntValue(lhs);
  auto rhsIntAttr = getConstantIntValue(rhs);

  // Both lhs and rhs are constants, return result directly.
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(
        std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // Otherwise, need to create instructions to calculate new attribute value.
  auto lhsValue = castToIndexType(builder, loc, lhs);
  auto rhsValue = castToIndexType(builder, loc, rhs);
  return builder.create<arith::MinSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult triton::mulOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                             OpBuilder &builder) {
  auto lhsIntAttr = getConstantIntValue(lhs);
  auto rhsIntAttr = getConstantIntValue(rhs);

  // Shortcuts for special cases.
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIntAttr) {
    if (rhsIntAttr.value() == 0)
      return rhs;
    if (rhsIntAttr == 1)
      return lhs;
  }

  // Both lhs and rhs are constants.
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(lhsIntAttr.value() * rhsIntAttr.value());

  // Otherwise, need to create instructions to calculate new attribute value.
  auto lhsValue = castToIndexType(builder, loc, lhs);
  auto rhsValue = castToIndexType(builder, loc, rhs);
  return builder.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult triton::maxOFRs(OpFoldResult lhs, OpFoldResult rhs, Location loc,
                             OpBuilder &builder) {
  auto lhsIntAttr = getConstantIntValue(lhs);
  auto rhsIntAttr = getConstantIntValue(rhs);

  // Both lhs and rhs are constants, return result directly.
  if (lhsIntAttr && rhsIntAttr)
    return builder.getIndexAttr(
        std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // Otherwise, need to create instructions to calculate new attribute value.
  auto lhsValue = castToIndexType(builder, loc, lhs);
  auto rhsValue = castToIndexType(builder, loc, rhs);
  return builder.create<arith::MaxSIOp>(loc, lhsValue, rhsValue).getResult();
}
