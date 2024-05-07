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

/// Derive the specific max/min semantics based on the type of compare and the
/// operand relationship between compare and select.
static std::optional<Operation *> getCmpSelectResult(PatternRewriter &rewriter,
                                                     Location loc,
                                                     arith::CmpFOp op,
                                                     bool operandsSwapped) {
  auto predicate = op.getPredicate();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  switch (predicate) {
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return operandsSwapped ? rewriter.create<arith::MinimumFOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MaximumFOp>(loc, lhs, rhs);
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return operandsSwapped ? rewriter.create<arith::MaximumFOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MinimumFOp>(loc, lhs, rhs);
  default:
    return std::nullopt;
  }
}

/// Derive the specific max/min semantics based on the type of compare and the
/// operand relationship between compare and select.
static std::optional<Operation *> getCmpSelectResult(PatternRewriter &rewriter,
                                                     Location loc,
                                                     arith::CmpIOp op,
                                                     bool operandsSwapped) {
  auto predicate = op.getPredicate();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  switch (predicate) {
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
    return operandsSwapped ? rewriter.create<arith::MinSIOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MaxSIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    return operandsSwapped ? rewriter.create<arith::MinUIOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MaxUIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
    return operandsSwapped ? rewriter.create<arith::MaxSIOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MinSIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    return operandsSwapped ? rewriter.create<arith::MaxUIOp>(loc, lhs, rhs)
                           : rewriter.create<arith::MinUIOp>(loc, lhs, rhs);
  default:
    return std::nullopt;
  }
}

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

    // Get cmp op mode.
    std::optional<arith::CmpFOp> cmpFOp;
    std::optional<arith::CmpIOp> cmpIOp;
    if (isa<arith::CmpFOp>(cmpOp)) {
      cmpFOp = cast<arith::CmpFOp>(cmpOp);
    } else {
      cmpIOp = cast<arith::CmpIOp>(cmpOp);
    }
    // Get specific max/min semantics.
    std::optional<Operation *> maxMinOp;
    auto loc = op.getLoc();
    if (op->getOperand(1) == cmpOp->getOperand(0) &&
        op->getOperand(2) == cmpOp->getOperand(1)) {
      if (cmpFOp) {
        maxMinOp = getCmpSelectResult(rewriter, loc, *cmpFOp, false);
      } else if (cmpIOp) {
        maxMinOp = getCmpSelectResult(rewriter, loc, *cmpIOp, false);
      }
    } else if (op->getOperand(1) == cmpOp->getOperand(1) &&
               op->getOperand(2) == cmpOp->getOperand(0)) {
      if (cmpFOp) {
        maxMinOp = getCmpSelectResult(rewriter, loc, *cmpFOp, true);
      } else if (cmpIOp) {
        maxMinOp = getCmpSelectResult(rewriter, loc, *cmpIOp, true);
      }
    }
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
                 CanonicalizeArithI1Pattern<arith::AddIOp, arith::XOrIOp>>(
        patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::arith_ext::createArithCanonicalizerPass() {
  return std::make_unique<ArithCanonicalizerPass>();
}
