//===- ArithToLinalg.cpp - Arith to Linalg conversion------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-linalg/Conversion/ArithToLinalg/ArithToLinalg.h"
#include "triton-linalg/Conversion/LinalgCommon/Pattern.h"
#include "triton-linalg/Conversion/PassDetail.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/Utils/ArithUtils.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::triton;
namespace mlir {
class MLIRContext;
} // namespace mlir

namespace {
class ArithConstantPattern : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {

    auto fillVal = getSplatValue(rewriter, op);
    if (failed(fillVal))
      return failure();

    auto loc = op.getLoc();
    auto resType = op.getType().cast<ShapedType>();
    Value init = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                  resType.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, *fillVal, init);
    return success();
  }
};

class ArithSelectPattern : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto condition = op.getCondition();
    auto trueValue = op.getTrueValue();
    auto falseValue = op.getFalseValue();

    if (!trueValue.getType().isa<ShapedType>() ||
        !falseValue.getType().isa<ShapedType>())
      return failure();

    auto initDims = getDims(rewriter, loc, trueValue);
    if (condition.getType().isSignlessInteger(1)) {
      Value initCondition = rewriter.create<tensor::EmptyOp>(
          loc, initDims, rewriter.getIntegerType(1));
      condition = rewriter.create<linalg::FillOp>(loc, condition, initCondition)
                      .getResult(0);
    }

    auto resElementType = getElementTypeOrSelf(op.getType());
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, initDims, resElementType);
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, ValueRange({condition, trueValue, falseValue}), initTensor,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value innerResult = b.create<arith::SelectOp>(loc, resElementType,
                                                        args, op->getAttrs());
          b.create<linalg::YieldOp>(loc, innerResult);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, mapOp->getResults());
    return success();
  }
};

} // namespace

void mlir::triton::populateArithToLinalgPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // Arith ops to linalg ops.
  patterns.add<
      // Constant op.
      ArithConstantPattern,
      // Fixing point ops.
      GenericOpPattern<arith::AddIOp>, GenericOpPattern<arith::SubIOp>,
      GenericOpPattern<arith::MulIOp>, GenericOpPattern<arith::DivUIOp>,
      GenericOpPattern<arith::DivSIOp>, GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>,
      // Floating point ops.
      GenericOpPattern<arith::NegFOp>, GenericOpPattern<arith::AddFOp>,
      GenericOpPattern<arith::SubFOp>,
      // MaxMin ops.
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxSIOp>,
      GenericOpPattern<arith::MaxUIOp>, GenericOpPattern<arith::MinimumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      GenericOpPattern<arith::MaxNumFOp>, GenericOpPattern<arith::MinNumFOp>,
      // Floating point ops.
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cast ops.
      GenericOpPattern<arith::IndexCastOp>, GenericOpPattern<arith::ExtUIOp>,
      GenericOpPattern<arith::ExtSIOp>, GenericOpPattern<arith::ExtFOp>,
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::UIToFPOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToUIOp>, GenericOpPattern<arith::FPToSIOp>,
      GenericOpPattern<arith::BitcastOp>,
      // Cmp ops.
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select op.
      ArithSelectPattern>(context);
}

namespace {
struct ArithToLinalgPass : public ArithToLinalgPassBase<ArithToLinalgPass> {
  ArithToLinalgPass() = default;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<linalg::LinalgDialect, linalg_ext::LinalgExtDialect,
                           tensor::TensorDialect>();
    target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *op) {
      return !op->getResultTypes().front().isa<ShapedType>();
    });
    // Setup conversion patterns.
    RewritePatternSet patterns(&ctx);
    populateArithToLinalgPatterns(patterns);
    // Apply conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createArithToLinalgPass() {
  return std::make_unique<ArithToLinalgPass>();
}
