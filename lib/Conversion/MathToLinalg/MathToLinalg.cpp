//===- MathToLinalg.cpp - Math to Linalg conversion------------*- C++ -*-===//
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-linalg/Conversion/LinalgCommon/Pattern.h"
#include "triton-linalg/Conversion/MathToLinalg/MathToLinalg.h"
#include "triton-linalg/Conversion/PassDetail.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/MathExt/IR/MathExt.h" // IWYU pragma: keep
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::triton;
namespace mlir {
class MLIRContext;
} // namespace mlir

void mlir::triton::populateMathToLinalgPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // Math ops to linalg ops.
  patterns.add<GenericOpPattern<math::LogOp>, GenericOpPattern<math::ExpOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::SqrtOp>, GenericOpPattern<math::AbsIOp>,
               GenericOpPattern<math::AbsFOp>, GenericOpPattern<math::TruncOp>,
               GenericOpPattern<math::RoundOp>, GenericOpPattern<math::FloorOp>,
               GenericOpPattern<math::FmaOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::Log2Op>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::ErfOp>,
               GenericOpPattern<math::TanhOp>,
               GenericOpPattern<math_ext::MulhiUIOp>>(context);
}

namespace {
struct MathToLinalgPass : public MathToLinalgPassBase<MathToLinalgPass> {
  MathToLinalgPass() = default;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ConversionTarget target(ctx);
    target.addLegalDialect<linalg::LinalgDialect, linalg_ext::LinalgExtDialect,
                           tensor::TensorDialect, arith::ArithDialect>();
    target.addDynamicallyLegalDialect<math::MathDialect,
                                      math_ext::MathExtDialect>(
        [&](Operation *op) {
          return !isa<ShapedType>(op->getResultTypes().front());
        });
    // Setup conversion patterns.
    RewritePatternSet patterns(&ctx);
    populateMathToLinalgPatterns(patterns);
    // Apply conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createMathToLinalgPass() {
  return std::make_unique<MathToLinalgPass>();
}
