//===- Pattern.h -[Generic pattern to linalg ops]----------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_CONVERSION_LINALGCOMMON_PATTERN_H
#define TRITON_LINALG_CONVERSION_LINALGCOMMON_PATTERN_H
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class Operation;
namespace triton {
template <class Op> class GenericOpPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Remain unchanged if one of operands is scalar.
    if (!llvm::all_of(adaptor.getOperands(),
                      [&](Value v) { return v.getType().isa<ShapedType>(); })) {
      return failure();
    }
    // Apply only if all operands are not scalar.
    auto loc = op.getLoc();
    auto resType = op.getType().template cast<ShapedType>();
    auto initDims = getDims(rewriter, loc, op->getOperand(0));
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, initDims, resType.getElementType());
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, adaptor.getOperands(), initTensor,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value innerResult = b.create<Op>(
              loc, getElementTypeOrSelf(op.getType()), args, op->getAttrs());
          b.create<linalg::YieldOp>(loc, innerResult);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, mapOp->getResults());
    return success();
  }
};
} // namespace triton
} // namespace mlir
#endif // TRITON_LINALG_CONVERSION_LINALGCOMMON_PATTERN_H
