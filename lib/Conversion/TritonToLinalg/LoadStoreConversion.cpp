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
    if (!op.getMask()) {
      sliceTensor = transformResultWithTransposeAndDimInfo(
          sliceTensor, ptrInfo->permutations, ptrInfo->dimInfos, ptrInfo->sizes,
          rewriter);
      rewriter.replaceOp(op, sliceTensor);
      return success();
    }

    sliceTensor = transformResultWithTransposeAndDimInfo(
        sliceTensor, ptrInfo->permutations, ptrInfo->dimInfos, ptrInfo->sizes,
        rewriter);

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
    auto mask = op.getMask();
    MaskTracker maskTracker;
    if (mask)
      (void)maskTracker.parse(mask, loc, rewriter);

    auto info = getPtrAndScatterInfo(
        loc, op.getPtr(), mask, maskTracker, resultTy, op.getOperation(),
        rewriter, getCacheModeAttr(op.getContext(), op.getCache()));
    if (failed(info))
      return failure();
    auto &[ptrInfo, scatteredInfo] = *info;
    Value sliceTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, ptrInfo.memref, true, true);

    SmallVector<Value> gatherInputs{sliceTensor, scatteredInfo.indice};
    if (scatteredInfo.mask)
      gatherInputs.push_back(scatteredInfo.mask);

    auto initShape = permutateAndRemoveBroadcastDims<OpFoldResult>(
        ptrInfo.sizes, ptrInfo.permutations, ptrInfo.dimInfos);

    // Reset the last windowRank - 1 dimensions to the size of original shape.
    if (scatteredInfo.windowRank > 1) {
      SmallVector<int64_t> originalShape = permutateAndRemoveBroadcastDims(
          resultTy.getShape(), ptrInfo.permutations, ptrInfo.dimInfos);
      int64_t rank = resultTy.getRank();
      for (auto i : llvm::seq(rank - scatteredInfo.windowRank + 1, rank))
        initShape[i] = rewriter.getIndexAttr(originalShape[i]);
    }

    Value init = rewriter.create<tensor::EmptyOp>(loc, initShape,
                                                  resultTy.getElementType());

    auto reshapedGatherResTy = init.getType().cast<RankedTensorType>();
    init = collapseLastNDimsToOneDim(rewriter, loc, init,
                                     scatteredInfo.windowRank);

    bool needAddUnitBatchDim =
        (reshapedGatherResTy.getRank() == scatteredInfo.windowRank);
    // Add an additional unit batch dim when batch dim number is 0.
    if (needAddUnitBatchDim) {
      gatherInputs[1] = prependUnitDim(rewriter, loc, gatherInputs[1]);
      // Add batch dim for mask if exists.
      if (gatherInputs.size() > 2)
        gatherInputs[2] = prependUnitDim(rewriter, loc, gatherInputs[2]);
      init = prependUnitDim(rewriter, loc, init);
    }

    // Do gather operation.
    Value gatherRes = rewriter
                          .create<linalg_ext::GatherOp>(
                              loc, gatherInputs, init,
                              /*dimensionMap=*/SmallVector<int64_t>({0}),
                              /*rangedData=*/false,
                              [](OpBuilder &b, Location loc, ValueRange args) {
                                b.create<linalg_ext::ExtYieldOp>(loc, args[0]);
                              })
                          .getResult()[0];
    // Remove the prepend unit batch dim.
    if (needAddUnitBatchDim)
      gatherRes = dropUnitFirstDim(rewriter, loc, gatherRes);

    gatherRes =
        reshapeGatherScatterValueTo(gatherRes, reshapedGatherResTy, rewriter);

    // Since the last windowRank - 1 dimensions may contain all elements, we
    // extract the masked part.
    if (scatteredInfo.windowRank > 1 && mask) {
      auto zeroAttr = rewriter.getIndexAttr(0);
      auto maskOffsets = permutateAndRemoveBroadcastDims<OpFoldResult>(
          maskTracker.getStarts(), ptrInfo.permutations, ptrInfo.dimInfos);
      std::for_each(
          maskOffsets.begin(), maskOffsets.end(),
          [zeroAttr](auto &ofr) { ofr = ofr ? ofr : OpFoldResult(zeroAttr); });

      auto maskSizes = permutateAndRemoveBroadcastDims<OpFoldResult>(
          ptrInfo.sizes, ptrInfo.permutations, ptrInfo.dimInfos);

      auto oneAttr = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> maskStrides(maskOffsets.size(), oneAttr);
      gatherRes = rewriter
                      .create<tensor::ExtractSliceOp>(
                          loc, gatherRes, maskOffsets, maskSizes, maskStrides)
                      .getResult();
    }

    auto res = postProcessGatheredResult(loc, gatherRes, mask, op.getOther(),
                                         ptrInfo.permutations, ptrInfo.dimInfos,
                                         ptrInfo.offsets, ptrInfo.sizes,
                                         resultTy, maskTracker, rewriter);

    rewriter.replaceOp(op, res);

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
    auto mask = op.getMask();
    MaskTracker maskTracker;
    if (mask)
      (void)maskTracker.parse(mask, loc, rewriter);

    auto info = getPtrAndScatterInfo(
        loc, op.getPtr(), mask, maskTracker, valueTy, op.getOperation(),
        rewriter, getCacheModeAttr(op.getContext(), op.getCache()));
    if (failed(info))
      return failure();
    auto [ptrInfo, scatteredInfo] = *info;

    Value sliceTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, ptrInfo.memref, true, true);

    // Get scatter update.
    Value update = getUpdateOperandForScatterOp(
        loc, op.getValue(), scatteredInfo.windowRank, ptrInfo, rewriter);
    SmallVector<Value> scatterInputs{update, scatteredInfo.indice};

    // Get scatter mask.
    if (scatteredInfo.mask)
      scatterInputs.push_back(scatteredInfo.mask);
    // Add an additional unit batch dim when batch dim number is 0.
    auto indiceTy = scatteredInfo.indice.getType().cast<RankedTensorType>();
    // The last dim of indice represents points of coordinates.
    if (indiceTy.getRank() == 1) {
      for (size_t i = 0; i < scatterInputs.size(); ++i)
        scatterInputs[i] = prependUnitDim(rewriter, loc, scatterInputs[i]);
    }

    Value scatterRes = rewriter
                           .create<linalg_ext::ScatterOp>(
                               loc, scatterInputs, sliceTensor,
                               SmallVector<int64_t>({0}), false, true,
                               [](OpBuilder &b, Location loc, ValueRange args) {
                                 b.create<linalg_ext::ExtYieldOp>(loc, args[0]);
                               })
                           ->getResult(0);

    rewriter.create<aux::StoreResourceOp>(loc, sliceTensor, scatterRes);
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

    TensorPointerMetaInfoTracker tracker;
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

    if (!op.getBoundaryCheck() || op.getBoundaryCheck()->empty()) {
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

    TensorPointerMetaInfoTracker tracker;
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
    if (op.getBoundaryCheck() && !op.getBoundaryCheck()->empty()) {
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
          converter, context, solver, 1);
  // Make gather/scatter pattern run at last.
  patterns
      .add<TritonScatteredLoadOpConversion, TritonScatteredStoreOpConversion>(
          converter, context, solver, 0);
  patterns
      .add<TritonTensorPtrLoadOpConversion, TritonTensorPtrStoreOpConversion>(
          converter, context);
}
