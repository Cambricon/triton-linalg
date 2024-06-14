//===- AtomicCASConversion.cpp - atomicCAS op conversion---------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/AtomicCASConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonPointerConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TypeConverter.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerMetaInfoTracker.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
class MLIRContext;
class DataFlowSolver;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonScalarAtomicCASOpConversion
    : public OpConversionPattern<triton::AtomicCASOp>,
      public triton::TritonPtrScalarConversionBase {
  using OpConversionPattern<triton::AtomicCASOp>::OpConversionPattern;

public:
  TritonScalarAtomicCASOpConversion(
      triton::TritonLinalgTypeConverter &converter, MLIRContext *context,
      DataFlowSolver &solver, PatternBenefit benefit)
      : OpConversionPattern<triton::AtomicCASOp>(converter, context, benefit),
        triton::TritonPtrScalarConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultTy =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    if (resultTy)
      return failure();

    auto loc = op.getLoc();
    auto elementType = op.getResult().getType();
    auto memref = getMemRef(loc, op.getPtr(), elementType, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref, true, true);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    RankedTensorType originTensorTy =
        originTensor.getType().cast<RankedTensorType>();

    auto cmpInit = rewriter.create<tensor::EmptyOp>(
        loc, originTensorTy.getShape(), op.getCmp().getType());
    Value cmpValue = rewriter.create<tensor::InsertOp>(
        loc, op.getCmp(), cmpInit, ValueRange({zero}));

    auto valInit = rewriter.create<tensor::EmptyOp>(
        loc, originTensorTy.getShape(), op.getVal().getType());
    Value valValue = rewriter.create<tensor::InsertOp>(
        loc, op.getVal(), valInit, ValueRange({zero}));

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, originTensorTy.getShape(), originTensorTy.getElementType());

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    auto casOp = rewriter.create<triton::linalg_ext::AtomicCASOp>(
        loc, originTensor.getType(),
        ValueRange({originTensor, cmpValue, valValue}), init,
        *maybeMemoryOrder);
    Value scalarRet =
        rewriter
            .create<tensor::ExtractOp>(loc, op.getResult().getType(),
                                       casOp->getResult(0), ValueRange({zero}))
            .getResult();
    op.replaceAllUsesWith(scalarRet);
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonAtomicCASPattern
    : public OpConversionPattern<triton::AtomicCASOp>,
      public triton::TritonPtrContiguousConversionBase {
public:
  TritonAtomicCASPattern(triton::TritonLinalgTypeConverter &converter,
                         MLIRContext *context, DataFlowSolver &solver,
                         PatternBenefit benefit = 1)
      : OpConversionPattern<triton::AtomicCASOp>(converter, context, benefit),
        triton::TritonPtrContiguousConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If atomic_cas on scalar type.
    RankedTensorType resultTy =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy)
      return failure();

    auto loc = op.getLoc();
    auto ptrInfo = getPtrInfo(loc, op.getPtr(), nullptr, resultTy, rewriter);
    if (failed(ptrInfo))
      return failure();
    // Atomic does not support broadcast and permutations.
    for (auto dimInfo : ptrInfo->dimInfos) {
      if (dimInfo.getContigSize() != dimInfo.getDimSize())
        return failure();
    }
    if (!triton::isConsecutive(ptrInfo->permutations)) {
      return failure();
    }

    Value originTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, ptrInfo->memref, true, true);

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                 resultTy.getElementType());

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    Value ret = rewriter
                    .create<triton::linalg_ext::AtomicCASOp>(
                        loc, op.getResult().getType(),
                        ValueRange({originTensor, op.getCmp(), op.getVal()}),
                        init, *maybeMemoryOrder)
                    .getResults()[0];
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class TritonGatherAtomicCASPattern
    : public OpConversionPattern<triton::AtomicCASOp>,
      TritonPtrScatterConversionBase {
public:
  TritonGatherAtomicCASPattern(triton::TritonLinalgTypeConverter &converter,
                               MLIRContext *context, DataFlowSolver &solver,
                               PatternBenefit benefit = 1)
      : OpConversionPattern<triton::AtomicCASOp>(converter, context, benefit),
        TritonPtrScatterConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If atomic_cas on scalar type.
    RankedTensorType resultTy =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy) {
      return failure();
    }

    auto loc = op.getLoc();
    // When encounter discrete input.
    triton::PointerMetaInfoTracker tracker;
    if (failed(tracker.parse(op.getPtr(), loc, rewriter)))
      return failure();
    Value memref = getDynamicMemRef(loc, tracker.getBase(), resultTy, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref, true, true);
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                 resultTy.getElementType());

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    rewriter.replaceOpWithNewOp<triton::linalg_ext::GatherAtomicCASOp>(
        op, op.getResult().getType(),
        ValueRange(
            {originTensor, op.getCmp(), op.getVal(), tracker.getOffset()}),
        init, *maybeMemoryOrder);
    return success();
  }
};

} // namespace

void triton::populateTritonAtomicCASToLinalgPatterns(
    RewritePatternSet &patterns, triton::TritonLinalgTypeConverter &converter,
    DataFlowSolver &solver) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonScalarAtomicCASOpConversion, TritonAtomicCASPattern>(
      converter, context, solver, 1);
  // Make gather/scatter pattern run at last.
  patterns.add<TritonGatherAtomicCASPattern>(converter, context, solver, 0);
}
