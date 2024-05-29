//===- LoadStoreConversion.cpp - Load/store op convension -------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <optional>
#include <stddef.h>
#include <stdint.h>

#include "triton-linalg/Conversion/TritonToLinalg/LoadStoreConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonPointerConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TypeConverter.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerMetaInfoTracker.h"
#include "triton-linalg/Dialect/Utils/Conventions.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"

namespace mlir {
class MLIRContext;
class DataFlowSolver;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {
class TritonContiguousLoadOpConversion
    : public OpConversionPattern<triton::LoadOp>,
      public TritonPtrContiguousConversionBase {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

public:
  TritonContiguousLoadOpConversion(TritonLinalgTypeConverter &converter,
                                   MLIRContext *context, DataFlowSolver &solver,
                                   PatternBenefit benefit)
      : OpConversionPattern<triton::LoadOp>(converter, context, benefit),
        TritonPtrContiguousConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::isTensorPointerType(op.getPtr().getType()))
      return failure();

    RankedTensorType resultTy =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy)
      return failure();

    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    bool allowMaskTrackFailureIgnore = isLinearMemory(module);
    auto ptrInfo =
        getPtrInfo(loc, op.getPtr(), op.getMask(), resultTy, rewriter,
                   getCacheModeAttr(op.getContext(), op.getCache()),
                   allowMaskTrackFailureIgnore);
    if (failed(ptrInfo)) {
      return failure();
    }

    Value sliceTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, ptrInfo->memref, true, true);
    auto tensorType = sliceTensor.getType().cast<RankedTensorType>();
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType(),
        getDynamicDimsValue(rewriter, loc, sliceTensor));
    sliceTensor = rewriter.create<linalg::CopyOp>(loc, sliceTensor, emptyTensor)
                      .getResultTensors()[0];
    sliceTensor = transformResultWithTransposeAndDimInfo(
        sliceTensor, ptrInfo->permutations, ptrInfo->dimInfos, ptrInfo->sizes,
        rewriter);
    if (!op.getMask()) {
      rewriter.replaceOp(op, sliceTensor);
      return success();
    }
    sliceTensor = getPadOrInsertOpWithOther(
        loc, op.getOther(),
        RankedTensorType::get(resultTy.getShape(), resultTy.getElementType()),
        sliceTensor, ptrInfo->offsets, ptrInfo->sizes, rewriter);

    if (ptrInfo->isMaskTrackerFailed && op.getOther()) {
      sliceTensor =
          selectByMask(loc, op.getMask(), sliceTensor, op.getOther(), rewriter);
    }
    rewriter.replaceOp(op, sliceTensor);
    return success();
  }
};

class TritonContiguousStoreOpConversion
    : public OpConversionPattern<triton::StoreOp>,
      public TritonPtrContiguousConversionBase {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

public:
  TritonContiguousStoreOpConversion(TritonLinalgTypeConverter &converter,
                                    MLIRContext *context,
                                    DataFlowSolver &solver,
                                    PatternBenefit benefit)
      : OpConversionPattern<triton::StoreOp>(converter, context, benefit),
        TritonPtrContiguousConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::isTensorPointerType(op.getPtr().getType()))
      return failure();

    RankedTensorType valueTy =
        op.getValue().getType().dyn_cast<RankedTensorType>();

    if (!valueTy)
      return failure();

    auto loc = op.getLoc();
    auto ptrInfo = getPtrInfo(loc, op.getPtr(), op.getMask(), valueTy, rewriter,
                              getCacheModeAttr(op.getContext(), op.getCache()));
    if (failed(ptrInfo))
      return failure();

    Value value = op.getValue();
    auto zeroAttr = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> defaultOffsets(valueTy.getRank(), zeroAttr);
    value = transformInputWithTransposeAndDimInfo(value, ptrInfo->permutations,
                                                  ptrInfo->dimInfos,
                                                  defaultOffsets, rewriter);
    if (op.getMask()) {
      auto rank = value.getType().cast<ShapedType>().getRank();
      value = rewriter.create<tensor::ExtractSliceOp>(
          loc, value,
          permutateAndRemoveBroadcastDims<OpFoldResult>(
              ptrInfo->offsets, ptrInfo->permutations, ptrInfo->dimInfos),
          permutateAndRemoveBroadcastDims<OpFoldResult>(
              ptrInfo->sizes, ptrInfo->permutations, ptrInfo->dimInfos),
          SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));
    }
    auto materializeOp =
        rewriter.create<bufferization::MaterializeInDestinationOp>(
            op.getLoc(), value, ptrInfo->memref);
    materializeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonScalarLoadOpConversion : public OpConversionPattern<triton::LoadOp>,
                                     public TritonPtrScalarConversionBase {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

public:
  TritonScalarLoadOpConversion(TritonLinalgTypeConverter &converter,
                               MLIRContext *context, DataFlowSolver &solver,
                               PatternBenefit benefit)
      : OpConversionPattern<triton::LoadOp>(converter, context, benefit),
        TritonPtrScalarConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getResult().getType().dyn_cast<RankedTensorType>())
      return failure();

    auto loc = op.getLoc();
    auto elementType = op.getResult().getType();
    auto memref = getMemRef(loc, op.getPtr(), elementType, rewriter,
                            getCacheModeAttr(op.getContext(), op.getCache()));
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value scalar =
        rewriter.create<memref::LoadOp>(loc, elementType, memref, c0);
    if (!op.getMask()) {
      rewriter.replaceOp(op, scalar);
      return success();
    }

    auto ifOp = rewriter.create<scf::IfOp>(
        loc, op.getMask(),
        [scalar](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, scalar);
        },
        [other = adaptor.getOther(), scalar](OpBuilder &b, Location loc) {
          // If other is nullptr, it's a undefined case, here we just yield
          // original scalar.
          if (!other) {
            b.create<scf::YieldOp>(loc, scalar);
          } else {
            b.create<scf::YieldOp>(loc, other);
          }
        });
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

class TritonScalarStoreOpConversion
    : public OpConversionPattern<triton::StoreOp>,
      public TritonPtrScalarConversionBase {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

public:
  TritonScalarStoreOpConversion(TritonLinalgTypeConverter &converter,
                                MLIRContext *context, DataFlowSolver &solver,
                                PatternBenefit benefit)
      : OpConversionPattern<triton::StoreOp>(converter, context, benefit),
        TritonPtrScalarConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getValue().getType().dyn_cast<RankedTensorType>())
      return failure();

    auto loc = op.getLoc();
    auto memref = getMemRef(loc, op.getPtr(), op.getValue().getType(), rewriter,
                            getCacheModeAttr(op.getContext(), op.getCache()));

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, op.getValue(), memref, c0);
    rewriter.eraseOp(op);
    return success();
  }
};

class TritonScatteredLoadOpConversion
    : public OpConversionPattern<triton::LoadOp>,
      TritonPtrScatterConversionBase {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

public:
  TritonScatteredLoadOpConversion(TritonLinalgTypeConverter &converter,
                                  MLIRContext *context, DataFlowSolver &solver,
                                  PatternBenefit benefit)
      : OpConversionPattern<triton::LoadOp>(converter, context, benefit),
        TritonPtrScatterConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::isTensorPointerType(op.getPtr().getType()))
      return failure();
    auto resultTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy)
      return failure();

    auto loc = op.getLoc();
    PointerMetaInfoTracker tracker;
    if (failed(tracker.parse(op.getPtr(), loc, rewriter)))
      return failure();
    Value memref = getDynamicMemRef(loc, tracker.getBase(), resultTy, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref, true, true);

    // Get window.
    auto window = op.getOther();
    if (!window) {
      window = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), resultTy.getShape(), resultTy.getElementType());
    }
    window = flattenValueToMatchGatherScatter(rewriter, window);

    // Get value.
    Value indices =
        flattenValueToMatchGatherScatter(rewriter, tracker.getOffset());

    SmallVector<Value> gatherInputs{originTensor, indices};

    // Get mask.
    if (op.getMask()) {
      auto mask =
          flattenValueToMatchGatherScatter(rewriter, op.getMask(), false);
      gatherInputs.push_back(mask);
    }

    // Do gather operation.
    Value gatherRes = rewriter
                          .create<linalg_ext::GatherOp>(
                              loc, gatherInputs, window,
                              /*dimensionMap=*/SmallVector<int64_t>({0}),
                              /*rangedData=*/false,
                              [](OpBuilder &b, Location loc, ValueRange args) {
                                b.create<linalg_ext::ExtYieldOp>(loc, args[0]);
                              })
                          .getResult()[0];
    gatherRes = reshapeGatherScatterValueTo(gatherRes, resultTy, rewriter);
    rewriter.replaceOp(op, gatherRes.getDefiningOp()->getResults());

    return success();
  }
};

class TritonScatteredStoreOpConversion
    : public OpConversionPattern<triton::StoreOp>,
      TritonPtrScatterConversionBase {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

public:
  TritonScatteredStoreOpConversion(TritonLinalgTypeConverter &converter,
                                   MLIRContext *context, DataFlowSolver &solver,
                                   PatternBenefit benefit)
      : OpConversionPattern<triton::StoreOp>(converter, context, benefit),
        TritonPtrScatterConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::isTensorPointerType(op.getPtr().getType()))
      return failure();

    auto valueTy = op.getValue().getType().dyn_cast<RankedTensorType>();
    if (!valueTy)
      return failure();

    auto loc = op.getLoc();

    PointerMetaInfoTracker tracker;
    if (failed(tracker.parse(op.getPtr(), loc, rewriter)))
      return failure();

    Value memref = getDynamicMemRef(loc, tracker.getBase(), valueTy, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref);
    // Get scatter init.
    Value scatterInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), getDim(rewriter, loc, originTensor, 0),
        valueTy.getElementType());

    // Get indices and updates.
    Value indices =
        flattenValueToMatchGatherScatter(rewriter, tracker.getOffset());
    Value updates = flattenValueToMatchGatherScatter(rewriter, op.getValue());

    SmallVector<Value> scatterInputs{updates, indices};
    // Get mask.
    auto mask = op.getMask();
    if (mask) {
      mask = flattenValueToMatchGatherScatter(rewriter, mask, false);
      scatterInputs.push_back(mask);
    }

    Value scatterRes = rewriter
                           .create<linalg_ext::ScatterOp>(
                               loc, scatterInputs, scatterInit,
                               SmallVector<int64_t>({0}), false, true,
                               [](OpBuilder &b, Location loc, ValueRange args) {
                                 b.create<linalg_ext::ExtYieldOp>(loc, args[0]);
                               })
                           ->getResult(0);

    rewriter.create<aux::StoreResourceOp>(op.getLoc(), originTensor,
                                          scatterRes);
    rewriter.eraseOp(op);
    return success();
  }

private:
  Value
  getUpdateOperandForScatterOp(Location loc, Value originUpdate,
                               int64_t windowRank, PtrInfo &ptrInfo,
                               ConversionPatternRewriter &rewriter) const {
    auto zeroAttr = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> defaultOffsets(ptrInfo.dimInfos.size(), zeroAttr);
    Value update = transformInputWithTransposeAndDimInfo(
        originUpdate, ptrInfo.permutations, ptrInfo.dimInfos, defaultOffsets,
        rewriter);

    auto oneAttr = rewriter.getIndexAttr(1);
    auto transposedOffsets = permutateAndRemoveBroadcastDims<OpFoldResult>(
        ptrInfo.offsets, ptrInfo.permutations, ptrInfo.dimInfos);
    auto transposedSizes = permutateAndRemoveBroadcastDims<OpFoldResult>(
        ptrInfo.sizes, ptrInfo.permutations, ptrInfo.dimInfos);
    SmallVector<OpFoldResult> strides(transposedSizes.size(), oneAttr);
    update = rewriter.create<tensor::ExtractSliceOp>(
        loc, update, transposedOffsets, transposedSizes, strides);

    update = collapseLastNDimsToOneDim(rewriter, loc, update, windowRank);

    return update;
  }
};

//////////////////////////// TensorPtr ///////////////////////////////////////
/// Order is the reverse of permutation in linalg::Transpose.
static SmallVector<int64_t> getPermutationFromOrder(ArrayRef<int32_t> order) {
  SmallVector<int64_t> permutation(order.size(), 0);
  for (const auto &dim : llvm::enumerate(order)) {
    permutation[dim.index()] = order.size() - 1 - dim.value();
  }
  return permutation;
}

class TritonTensorPtrLoadOpConversion
    : public OpConversionPattern<triton::LoadOp>,
      public TritonTensorPtrLoadStoreOpConversionBase {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isTensorPointerType(op.getPtr().getType()))
      return failure();
    RankedTensorType resultTy =
        op.getResult().getType().cast<RankedTensorType>();
    auto loc = op.getLoc();
    if (op.getMask() || op.getOther())
      return rewriter.notifyMatchFailure(
          loc, "Load with tensor pointers do not have mask and other");

    triton::TensorPointerMetaInfoTracker tracker;
    if (tracker.parse(op.getPtr(), loc, rewriter).failed())
      return failure();
    SmallVector<int64_t> permutations =
        getPermutationFromOrder(tracker.getOrder());
    auto dimInfos = getDimInfos(tracker.getStrides(), resultTy.getShape());
    auto sizes = getActualSizes(loc, op.getBoundaryCheck(), resultTy.getShape(),
                                tracker, rewriter);
    auto originalMemRef =
        getMemRef(rewriter.getRemappedValue(tracker.getBase()),
                  tracker.getOffsets(), sizes, tracker.getStrides(),
                  permutations, dimInfos, resultTy.getElementType(), rewriter,
                  getCacheModeAttr(op.getContext(), op.getCache()));

    Value sliceTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, originalMemRef, true, true);
    auto tensorType = sliceTensor.getType().cast<RankedTensorType>();
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType(),
        getDynamicDimsValue(rewriter, loc, sliceTensor));
    sliceTensor = rewriter.create<linalg::CopyOp>(loc, sliceTensor, emptyTensor)
                      .getResultTensors()[0];

    if (op.getBoundaryCheck().empty()) {
      sliceTensor = transformResultWithTransposeAndDimInfo(
          sliceTensor, permutations, dimInfos, sizes, rewriter);
      rewriter.replaceOp(op, sliceTensor);
      return success();
    }

    Value other = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                   resultTy.getElementType());
    if (op.getPadding()) {
      auto elementType = resultTy.getElementType();
      // Set zero padding value.
      TypedAttr attr =
          elementType.isIntOrIndex()
              ? rewriter.getIntegerAttr(elementType, 0).cast<TypedAttr>()
              : rewriter.getFloatAttr(elementType, 0).cast<TypedAttr>();

      // Float NaN padding case.
      if (op.getPadding().value() == triton::PaddingOption::PAD_NAN) {
        assert(!elementType.isIntOrIndex());
        auto apNaN = llvm::APFloat::getNaN(
            attr.cast<FloatAttr>().getValue().getSemantics());
        attr = rewriter.getFloatAttr(elementType, apNaN);
      }
      other = rewriter.create<arith::ConstantOp>(loc, attr);
    }

    auto value = transformResultWithTransposeAndDimInfo(
        sliceTensor, permutations, dimInfos, sizes, rewriter);
    value = getPadOrInsertOpWithOther(
        loc, other, resultTy, value,
        SmallVector<OpFoldResult>(resultTy.getRank(), rewriter.getIndexAttr(0)),
        sizes, rewriter);
    rewriter.replaceOp(op, value);
    return success();
  }
};

class TritonTensorPtrStoreOpConversion
    : public OpConversionPattern<triton::StoreOp>,
      public TritonTensorPtrLoadStoreOpConversionBase {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::isTensorPointerType(op.getPtr().getType()))
      return failure();
    RankedTensorType valueTy = op.getValue().getType().cast<RankedTensorType>();
    auto loc = op.getLoc();
    if (op.getMask())
      return rewriter.notifyMatchFailure(
          loc, "Store with tensor pointers do not have mask");

    triton::TensorPointerMetaInfoTracker tracker;
    if (tracker.parse(op.getPtr(), loc, rewriter).failed())
      return failure();

    SmallVector<int64_t> permutations =
        getPermutationFromOrder(tracker.getOrder());
    auto dimInfos = getDimInfos(tracker.getStrides(), valueTy.getShape());
    auto sizes = getActualSizes(loc, op.getBoundaryCheck(), valueTy.getShape(),
                                tracker, rewriter);
    auto originalMemRef =
        getMemRef(rewriter.getRemappedValue(tracker.getBase()),
                  tracker.getOffsets(), sizes, tracker.getStrides(),
                  permutations, dimInfos, valueTy.getElementType(), rewriter,
                  getCacheModeAttr(op.getContext(), op.getCache()));
    auto value = op.getValue();
    auto zeroAttr = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> defaultOffsets(dimInfos.size(), zeroAttr);
    value = transformInputWithTransposeAndDimInfo(value, permutations, dimInfos,
                                                  defaultOffsets, rewriter);
    if (!op.getBoundaryCheck().empty()) {
      auto rank = value.getType().cast<ShapedType>().getRank();
      value = rewriter.create<tensor::ExtractSliceOp>(
          loc, value, SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0)),
          permutateAndRemoveBroadcastDims<OpFoldResult>(sizes, permutations,
                                                        dimInfos),
          SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));
    }
    auto materializeOp =
        rewriter.create<bufferization::MaterializeInDestinationOp>(
            op.getLoc(), value, originalMemRef);
    materializeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void triton::populateTritonLoadStoreToLinalgPatterns(
    RewritePatternSet &patterns, TritonLinalgTypeConverter &converter,
    DataFlowSolver &solver) {
  MLIRContext *context = patterns.getContext();
  patterns
      .add<TritonContiguousLoadOpConversion, TritonContiguousStoreOpConversion,
           TritonScalarLoadOpConversion, TritonScalarStoreOpConversion>(
          converter, context, solver, 100);
  // Make gather/scatter pattern run at last.
  patterns
      .add<TritonScatteredLoadOpConversion, TritonScatteredStoreOpConversion>(
          converter, context, solver, 0);
  patterns
      .add<TritonTensorPtrLoadOpConversion, TritonTensorPtrStoreOpConversion>(
          converter, context);
}
