//===- TritonToTensor.cpp - Triton to Tensor dialect convension -*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-linalg/Conversion/PassDetail.h"
#include "triton-linalg/Conversion/TritonToTensor/TritonToTensor.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

#define DEBUG_TYPE "triton-to-tensor"

using namespace mlir;
using namespace triton;

namespace {
struct ConvertCatToinsertSlicePattern
    : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getResult().getType().cast<RankedTensorType>();

    Location loc = op.getLoc();
    Value result = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultTy.getElementType());

    auto rank = resultTy.getRank();
    auto operands = adaptor.getOperands();
    // Insert slice params.
    auto zero = rewriter.getIndexAttr(0);
    auto one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets(rank, zero);
    SmallVector<OpFoldResult> strides(rank, one);
    SmallVector<OpFoldResult> sizes;

    for (auto operand : operands) {
      sizes = getDims(rewriter, loc, operand);
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, operand, result, offsets, sizes, strides);
      // Triton's cat op always concat the 1st axis.
      offsets[0] = rewriter.createOrFold<arith::AddIOp>(
          loc, materializeOpFoldResult(rewriter, loc, offsets[0]),
          materializeOpFoldResult(rewriter, loc, sizes[0]));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TritonToTensorPass : public TritonToTensorPassBase<TritonToTensorPass> {
  void runOnOperation() override;
};
} // namespace

void TritonToTensorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);

  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isa<triton::CatOp>(op); });

  // Rewrite patterns.
  RewritePatternSet patterns(context);
  patterns.add<ConvertCatToinsertSlicePattern>(context);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::triton::createTritonToTensorPass() {
  return std::make_unique<TritonToTensorPass>();
}
