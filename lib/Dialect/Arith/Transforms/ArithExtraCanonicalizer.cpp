//===- ArithExtraCanonicalizer.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// \Note: it's a supplement for original linalg canonicalization defined in
// mlir/lib/Dialect/Arithmetic/IR/ArithmeticOps.cpp.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include "triton-linalg/Dialect/Arith/Transforms/Passes.h"
#include "triton-linalg/Dialect/Utils/ArithUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
using namespace mlir;
using namespace mlir::triton;
namespace mlir {
class MLIRContext;
} // namespace mlir

namespace {
///
/// Convert "cst->div" to "1/cst->mul".
///
/// Example:
/// ```
///   %cst = arith.constant 4.0 : f32
///   %0 = arith.divf %arg0, %cst : f32
/// ```
///
/// transformed into:
///
/// ```
///   %cst = arith.constant 2.500000e-01 : f32
///   %0 = arith.mulf %arg0, %cst : f32
/// ```
struct ScalarDivToMul final : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto divisor = op.getRhs();
    auto loc = op.getLoc();

    // Case1: 'rhs' is a scalar.
    FloatAttr divisorAttr;
    if (matchPattern(divisor, m_Constant(&divisorAttr))) {
      auto divisorVal = divisorAttr.getValue().convertToDouble();
      Value multiplier = rewriter.create<arith::ConstantOp>(
          loc, FloatAttr::get(divisor.getType(), 1.0 / divisorVal));
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), multiplier);
      return success();
    }

    // Case2: 'rhs' is a const float tensor.
    auto constDivisor = divisor.getDefiningOp<arith::ConstantOp>();
    auto divisorType = divisor.getType().dyn_cast_or_null<TensorType>();
    if (!constDivisor || !divisorType ||
        !divisorType.getElementType().isa<FloatType>()) {
      return failure();
    }
    auto constAttr = constDivisor.getValue().dyn_cast<DenseElementsAttr>();
    // Take the reciprocal element by element.
    auto multiplierVal = llvm::to_vector(llvm::map_range(
        constAttr.getValues<APFloat>(), [&](const APFloat &value) -> Attribute {
          auto divisorVal = value.convertToDouble();
          return FloatAttr::get(divisorType.getElementType(), 1.0 / divisorVal);
        }));
    auto multiplierAttr = DenseElementsAttr::get(divisorType, multiplierVal);
    auto multiplier = rewriter.create<arith::ConstantOp>(loc, multiplierAttr);

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), multiplier);
    return success();
  }
};

/// Canonicalize arith cmp and select to arith max/min pattern.
///
/// Example:
/// ```
///   %0 = arith.cmpf ogt, %arg0, %arg1 : f32
///   %1 = arith.select %0, %arg0, %arg1 : f32
/// ```
///
/// transformed into:
///
/// ```
///   %0 = arith.maximumf %arg0, %arg1 : f32
/// ```
/// FIXME(anxinqi): wrong conversion in float type, fix in GENESIS-1238
struct CanonicalizeCmpSelectToMinMax final
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Check cmp op.
    auto *cmpOp = op.getCondition().getDefiningOp();
    if (!cmpOp || !isa<arith::CmpFOp, arith::CmpIOp>(cmpOp)) {
      return failure();
    }
    auto maxMinOp = getCmpSelectResult(rewriter, cmpOp, op); 
    if (!maxMinOp) {
      return mlir::failure();
    }

    rewriter.replaceOp(op, maxMinOp.value()->getResults());
    return success();
  }
};

template <typename Op, typename TargetOp>
class CanonicalizeArithI1Pattern : public OpRewritePattern<Op> {
public:
  explicit CanonicalizeArithI1Pattern(MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Type eleType = getElementTypeOrSelf(op.getLhs().getType());
    if (!eleType.isInteger(1)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

/// Check whether the graph is constructed by nan statement
/// operations and the cmp-select pattern can be optimized to `OpTy`.
/// The nan structure region as follows:
///             bArg0    bArg1
///               ||  \   /
///               ||   cmpf
///               ||     |
///              cmpf    |
///                 \   /
///                  ori bArg0 bArg1
///                   |  /    /
///                   select
template <typename OpTy>
class CanonicalizeNanStatement : public OpRewritePattern<arith::SelectOp> {
public:
  explicit CanonicalizeNanStatement(MLIRContext *ctx)
      : OpRewritePattern<arith::SelectOp>(ctx) {}

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    using mlir::matchers::m_Any;
    // Nan pattern match.
    auto nanStatementPattern = m_Op<arith::SelectOp>(
        m_Op<arith::OrIOp>(m_Op<arith::CmpFOp>(m_Any(), m_Any()),
                           m_Op<arith::CmpFOp>(m_Any(), m_Any())),
        m_Any(), m_Any());
    if (!nanStatementPattern.match(selectOp)) {
      return failure();
    }
    auto trueVal = selectOp.getTrueValue();
    auto falseVal = selectOp.getFalseValue();
    // Collect block arg cmp users.
    llvm::SmallSetVector<Operation *, 2> cmpOps;
    for (auto user = trueVal.getUsers().begin(); user != trueVal.getUsers().end();
         user++) {
      if (isa<arith::CmpFOp>(*user)) {
        cmpOps.insert(*user);
      }
    }
    for (auto user = falseVal.getUsers().begin();
         user != falseVal.getUsers().end(); user++) {
      if (isa<arith::CmpFOp>(*user)) {
        cmpOps.insert(*user);
      }
    }
    // Must be two cmp ops.
    if (cmpOps.size() != 2) {
      return failure();
    }
    // One of cmp op must be une predicate and has same operands.
    if (llvm::count_if(cmpOps, [](Operation *op) {
          auto nanCmp = cast<arith::CmpFOp>(op);
          return (nanCmp.getPredicate() == arith::CmpFPredicate::UNE) &&
                 (nanCmp.getLhs() == nanCmp.getRhs());
        }) != 1) {
      return failure();
    }
    // Find non-une cmp op.
    auto *minMaxCmp = llvm::find_if(cmpOps, [](Operation *op) {
      auto cmp = cast<arith::CmpFOp>(op);
      return cmp.getPredicate() != arith::CmpFPredicate::UNE;
    });
    // Check if cmp and select can be optimized to min/max.
    auto minMaxOp = getCmpSelectResult(rewriter, *minMaxCmp, selectOp);
    if (!minMaxOp.has_value() || !isa_and_nonnull<OpTy>(minMaxOp.value())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<OpTy>(selectOp, trueVal, falseVal);
    return success();
  }
};

struct ArithCanonicalizerPass
    : public arith_ext::ArithCanonicalizerBase<ArithCanonicalizerPass> {
  ArithCanonicalizerPass() = default;
  ArithCanonicalizerPass(const ArithCanonicalizerPass &) = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    auto *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ScalarDivToMul, CanonicalizeCmpSelectToMinMax,
                 CanonicalizeArithI1Pattern<arith::MulIOp, arith::AndIOp>,
                 CanonicalizeArithI1Pattern<arith::AddIOp, arith::XOrIOp>,
		 CanonicalizeNanStatement<arith::MaximumFOp>,
		 CanonicalizeNanStatement<arith::MinimumFOp>>(
        patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::arith_ext::createArithCanonicalizerPass() {
  return std::make_unique<ArithCanonicalizerPass>();
}
