//===- TritonPointerConversion.cpp - pointer conversion ---------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <limits>
#include <optional>
#include <stddef.h>
#include <stdint.h>

#include "triton-linalg/Analysis/AxisInfoAnalysis.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonPointerConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/Triton/Interfaces/InferAxisInfoInterface.h"
#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Utils/Utils.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerMetaInfoTracker.h"
#include "triton-linalg/Dialect/Utils/Conventions.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::triton;

Value triton::selectByMask(Location loc, Value mask, Value trueVal,
                            Value falseVal,
                            ConversionPatternRewriter &rewriter) {
  assert(trueVal && "Get true value failed.");
  auto trueType = trueVal.getType().dyn_cast<ShapedType>();
  if (!mask || !falseVal || !trueType)
    return trueVal;

  auto falseType = falseVal.getType();
  if (!falseType.isa<ShapedType>()) {
    Value falseValInit =
        rewriter.create<tensor::EmptyOp>(loc, trueType.getShape(), falseType);
    falseVal =
        rewriter.create<linalg::FillOp>(loc, falseVal, ValueRange{falseValInit})
            .getResult(0);
  }
  auto resType = falseType.template cast<ShapedType>();
  auto initDims = triton::getDims(rewriter, loc, falseVal);
  Value initTensor =
      rewriter.create<tensor::EmptyOp>(loc, initDims, resType.getElementType());
  auto mapOp = rewriter.create<linalg::MapOp>(
      loc, ValueRange{mask, trueVal, falseVal}, initTensor,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value innerResult =
            b.create<arith::SelectOp>(loc, getElementTypeOrSelf(resType), args);
        b.create<linalg::YieldOp>(loc, innerResult);
      });
  return mapOp->getResult(0);
}

Value triton::flattenValueToMatchGatherScatter(
    ConversionPatternRewriter &rewriter, Value value, bool appendUnitDim) {
  if (!value)
    return value;

  auto valueTy = value.getType().cast<RankedTensorType>();
  auto loc = value.getLoc();
  auto rank = valueTy.getRank();

  if (rank == 0) {
    // Zero rank.
    SmallVector<ReassociationIndices> expandReassociation({});
    auto newType = RankedTensorType::get({1}, valueTy.getElementType());
    if (appendUnitDim) {
      newType = RankedTensorType::get({1, 1}, valueTy.getElementType());
    }
    return rewriter.create<tensor::ExpandShapeOp>(loc, newType, value,
                                                  expandReassociation);
  }

  if (rank > 1) {
    auto seq = llvm::seq<int64_t>(0, rank);
    SmallVector<ReassociationIndices> collapseReassociation(
        {{seq.begin(), seq.end()}});
    value = rewriter.create<tensor::CollapseShapeOp>(loc, value,
                                                     collapseReassociation);
  }

  if (!appendUnitDim)
    return value;

  SmallVector<ReassociationIndices> expandReassociation({{0, 1}});
  return rewriter.create<tensor::ExpandShapeOp>(
      value.getLoc(),
      RankedTensorType::get({valueTy.getNumElements(), 1},
                            valueTy.getElementType()),
      value, expandReassociation);
}

Value triton::reshapeGatherScatterValueTo(
    Value value, RankedTensorType resultTy,
    ConversionPatternRewriter &rewriter) {
  assert(value);
  auto valueTy = value.getType().cast<RankedTensorType>();
  auto loc = value.getLoc();
  auto dstRank = resultTy.getRank();
  auto srcRank = value.getType().cast<RankedTensorType>().getRank();

  if (dstRank == 0) {
    // Zero rank.
    SmallVector<ReassociationIndices> collapseReassociation({});
    auto newType = RankedTensorType::get({}, valueTy.getElementType());
    return rewriter.create<tensor::CollapseShapeOp>(loc, newType, value,
                                                    collapseReassociation);
  }

  // Keep the first `srcRank - 1` dims unchanged, and reshape the last dim
  // to resultShape[srcRank - 1:dstRank].
  if (srcRank > dstRank) {
    assert(srcRank == dstRank + 1 &&
           "Unsupport case with source rank larger than `dest rank + 1`.");
    SmallVector<ReassociationIndices> collapseReassociation(dstRank);
    for (int64_t i = 0; i < dstRank; ++i)
      collapseReassociation[i] = {i};
    collapseReassociation.back().push_back(dstRank);
    return rewriter.create<tensor::CollapseShapeOp>(loc, value,
                                                    collapseReassociation);
  }

  if (srcRank < dstRank) {
    SmallVector<ReassociationIndices> expandReassociation(srcRank);
    for (int64_t i = 0; i < srcRank; ++i)
      expandReassociation[i] = {i};
    for (int64_t i = srcRank; i < dstRank; ++i)
      expandReassociation.back().push_back(i);
    return rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, value,
                                                  expandReassociation);
  }

  return value;
}

//===----------------------------------------------------------------------===//
// TritonPtrConversionBase
//===----------------------------------------------------------------------===//

Value TritonPtrConversionBase::getMemRef(
    Value base, OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<int64_t> permutations,
    ArrayRef<DimInfo> dimInfos, Type elementType,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode) const {
  auto llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  assert(sizes.size() == strides.size() &&
         (sizes.size() == permutations.size() || permutations.empty()) &&
         (sizes.size() == dimInfos.size() || dimInfos.empty()));

  auto newStrides =
      permutateAndRemoveBroadcastDims(strides, permutations, dimInfos);
  auto newSizes =
      permutateAndRemoveBroadcastDims(sizes, permutations, dimInfos);

  Location loc = base.getLoc();
  Value llvmPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, base);
  if (!offset)
    offset = rewriter.getIndexAttr(0);

  return rewriter.create<triton::aux::ViewOp>(
      loc, elementType, llvmPtr, offset, newSizes, newStrides, cacheMode);
}

Value TritonPtrConversionBase::getMemRef(
    Value base, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<int64_t> permutations,
    ArrayRef<DimInfo> dimInfos, Type elementType,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode) const {
  assert(sizes.size() == offsets.size() && sizes.size() == strides.size() &&
         sizes.size() == permutations.size() &&
         sizes.size() == dimInfos.size());

  Location loc = base.getLoc();
  // Linerized offset.
  // finalOffset = baseOffset(0) + sum(offsets#i * strides#i)
  OpFoldResult finalOffset = rewriter.getIndexAttr(0);
  for (auto &&[offset, stride] : llvm::zip(offsets, strides)) {
    OpFoldResult multipy = mulOFRs(stride, offset, loc, rewriter);
    finalOffset = addOFRs(finalOffset, multipy, loc, rewriter);
  }

  return getMemRef(base, finalOffset, sizes, strides, permutations, dimInfos,
                   elementType, rewriter, cacheMode);
}

Value TritonPtrConversionBase::transformResultWithTransposeAndDimInfo(
    Value value, ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
    ArrayRef<OpFoldResult> actualSizes,
    ConversionPatternRewriter &rewriter) const {
  auto valueTy = value.getType().cast<ShapedType>();
  auto loc = value.getLoc();

  // As the shape of value has been transposed before broadcasted by dimInfos,
  // so we should get the transposedDimInfos.
  // For example, given:
  // ```
  // originShape = (a, b, c)
  // permutations = (1, 2, 0)
  // dimInfos = (broadcast, contig, contig)
  // ```
  // In the getMemRef function, the following transformation will be performed.
  // ```
  // after transpose: (b, c, a)
  // after diminfo: (b, c)
  // ```
  //
  // So here, we revert the transformation using linalg.tranpose and
  // linalg.broadcast ops. then
  // ```
  // originShape = (b, c)
  // after broadcast: (b, c, a)
  // after transpose: (a, b, c)
  // ```
  auto transposedDimInfos = getValuesByPerms(dimInfos, permutations);
  // Get broadcast dimensions.
  SmallVector<int64_t> broadcastDimensions;
  SmallVector<OpFoldResult> broadcastShape =
      getValuesByPerms(actualSizes, permutations);
  for (const auto &dimInfo : llvm::enumerate(transposedDimInfos)) {
    assert(dimInfo.value().getBroadcastSize() == 1 ||
           dimInfo.value().getContigSize() == 1 ||
           dimInfo.value().getKind() == DimInfo::Kind::OTHERS);
    if (dimInfo.value().isBroadcastDim()) {
      broadcastDimensions.push_back(dimInfo.index());
    }
  }

  Value ret = value;
  if (!broadcastDimensions.empty()) {
    Value init = rewriter.create<tensor::EmptyOp>(loc, broadcastShape,
                                                  valueTy.getElementType());
    ret =
        rewriter
            .create<linalg::BroadcastOp>(loc, value, init, broadcastDimensions)
            .getResult()[0];
  }

  if (!isConsecutive(permutations)) {
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, getValuesByPerms<OpFoldResult>(broadcastShape, permutations),
        valueTy.getElementType());
    ret = rewriter.create<linalg::TransposeOp>(loc, ret, init, permutations)
              .getResult()[0];
  }
  return ret;
}

Value TritonPtrConversionBase::transformInputWithTransposeAndDimInfo(
    Value value, ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
    ArrayRef<OpFoldResult> offsets, ConversionPatternRewriter &rewriter) const {
  auto valueTy = value.getType().cast<ShapedType>();
  assert((!ShapedType::isDynamicShape(valueTy.getShape())) &&
         "value shape should be static");
  auto loc = value.getLoc();

  // As the shape of value has been transposed before broadcasted by dimInfos,
  // so we should get the transposedDimInfos.
  // For example, given:
  // ```
  // originShape = (a, b, c)
  // permutations = (1, 2, 0)
  // dimInfos = (broadcast, contig, contig)
  // ```
  // In the getMemRef function, the following transformation will be performed.
  // ```
  // after transpose: (b, c, a)
  // after dimInfo: (b, c)
  // ```
  Value ret = value;
  if (!isConsecutive(permutations)) {
    Value init = rewriter.create<tensor::EmptyOp>(
        loc,
        getValuesByPerms(ret.getType().cast<ShapedType>().getShape(),
                         permutations),
        valueTy.getElementType());
    ret = rewriter.create<linalg::TransposeOp>(loc, ret, init, permutations)
              .getResult()[0];
  }

  auto transposedDimInfos = getValuesByPerms(dimInfos, permutations);
  SmallVector<OpFoldResult> newSizes;
  newSizes.reserve(dimInfos.size());
  bool needExtractSlice = false;
  unsigned desiredResultRank = 0;
  for (const auto &dimInfo : llvm::enumerate(transposedDimInfos)) {
    assert(dimInfo.value().getBroadcastSize() == 1 ||
           dimInfo.value().getContigSize() == 1 ||
           dimInfo.value().getKind() == DimInfo::Kind::OTHERS);
    if (dimInfo.value().isBroadcastDim()) {
      needExtractSlice = true;
      newSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      newSizes.push_back(rewriter.getIndexAttr(dimInfo.value().getDimSize()));
      desiredResultRank++;
    }
  }
  if (!needExtractSlice)
    return ret;

  auto transposedOffsets =
      getValuesByPerms<OpFoldResult>(offsets, permutations);

  SmallVector<OpFoldResult> strides(dimInfos.size(), rewriter.getIndexAttr(1));
  /// Deduce the type of the result to use for the canonicalized operation.
  RankedTensorType resultType =
      tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
          desiredResultRank, ret.getType().cast<RankedTensorType>(),
          transposedOffsets, newSizes, strides);
  ret = rewriter
            .create<tensor::ExtractSliceOp>(
                loc, resultType, ret, transposedOffsets, newSizes, strides)
            .getResult();
  return ret;
}

//===----------------------------------------------------------------------===//
// TritonPtrLoadStoreOpConversionBase
//===----------------------------------------------------------------------===//
const triton::AxisInfoExt *
TritonPtrLoadStoreOpConversionBase::getAxisInfo(Value ptr) const {
  auto *lattice = solver.lookupState<AxisInfoLattice>(ptr);
  if (!lattice)
    return nullptr;
  return &lattice->getValue();
}

SmallVector<OpFoldResult> TritonPtrLoadStoreOpConversionBase::getMaskedOffsets(
    int64_t rank, std::optional<MaskTracker> maskTracker,
    ConversionPatternRewriter &rewriter) const {
  auto zeroAttr = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> offsets(rank, zeroAttr);
  if (maskTracker)
    offsets = SmallVector<OpFoldResult>(maskTracker->getStarts());
  // Reassign start point which is failed to analyze to 0.
  std::for_each(offsets.begin(), offsets.end(), [zeroAttr](auto &ofr) {
    ofr = ofr ? ofr : OpFoldResult(zeroAttr);
  });
  return offsets;
}

SmallVector<DimInfo> TritonPtrLoadStoreOpConversionBase::getDimInfos(
    const triton::AxisInfoExt *axisInfo, ArrayRef<int64_t> tensorShape) const {
  SmallVector<DimInfo> dimInfos;
  dimInfos.reserve(tensorShape.size());
  for (const auto &dim : llvm::enumerate(tensorShape)) {
    if (axisInfo->isConstantDim(tensorShape, dim.index())) {
      dimInfos.push_back({1, dim.value(), DimInfo::Kind::BROADCAST});
    } else if (axisInfo->isStrideDim(tensorShape, dim.index())) {
      dimInfos.push_back({dim.value(), 1, DimInfo::Kind::CONTIG});
    } else {
      dimInfos.push_back({dim.value()});
    }
  }
  return dimInfos;
}

SmallVector<int64_t> TritonPtrLoadStoreOpConversionBase::getPermutations(
    const triton::AxisInfoExt *axisInfo, ArrayRef<int64_t> tensorShape) const {
  // Switch the contiguous dim to the last dim.
  auto rank = axisInfo->getRank();
  assert(rank == tensorShape.size());
  SmallVector<int64_t> permutations =
      llvm::to_vector<2>(llvm::seq<int64_t>(0, rank));

  for (int64_t i = rank - 2; i >= 0; i--) {
    if (!axisInfo->isContiguousDim(tensorShape, rank - 1) &&
        axisInfo->isContiguousDim(tensorShape, i) && tensorShape[i] != 1) {
      std::swap(permutations[i], permutations[rank - 1]);
      break;
    }
  }
  return permutations;
}

SmallVector<OpFoldResult> TritonPtrLoadStoreOpConversionBase::getActualSizes(
    Location loc, ArrayRef<int64_t> tensorShape,
    std::optional<MaskTracker> maskTracker,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<OpFoldResult> blockSizes = llvm::to_vector<4>(
      llvm::map_range(tensorShape, [&rewriter](int64_t dim) -> OpFoldResult {
        return rewriter.getIndexAttr(dim);
      }));
  if (maskTracker) {
    assert(maskTracker->getRank() == tensorShape.size());
    for (const auto &dim : llvm::enumerate(maskTracker->getSizes())) {
      if (dim.value()) {
        blockSizes[dim.index()] = dim.value();
      }
    }
  }
  return blockSizes;
}

bool TritonPtrLoadStoreOpConversionBase::isBlockPtr(
    ArrayRef<DimInfo> dimInfos) const {
  return llvm::all_of(dimInfos, [](DimInfo dimInfo) {
    return dimInfo.getKind() != DimInfo::Kind::OTHERS;
  });
}

//===----------------------------------------------------------------------===//
// TritonPtrContiguousConversionBase
//===----------------------------------------------------------------------===//
std::pair<Value, SmallVector<Value>>
TritonPtrContiguousConversionBase::getOffsetAndStrides(
    Location loc, ArrayRef<int64_t> tensorShape, Value offset,
    std::optional<MaskTracker> maskTracker, const triton::AxisInfoExt *axisInfo,
    ConversionPatternRewriter &rewriter) const {
  size_t rank = tensorShape.size();
  // The strides:
  //    stride[0] = offset[1, 0, 0] - offset[0, 0, 0]
  //    stride[1] = offset[0, 1, 0] - offset[0, 0, 0]
  //    stride[2] = offset[0, 0, 1] - offset[0, 0, 0]
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> startIndices(rank, c0);
  if (maskTracker) {
    startIndices = vector::getAsValues(rewriter, loc, maskTracker->getStarts());
  }

  Value scalarOffset =
      rewriter.create<tensor::ExtractOp>(loc, offset, startIndices);

  assert(axisInfo->getRank() == rank);
  SmallVector<Value> strides(rank);
  for (size_t i = 0; i < rank; i++) {
    if (tensorShape[i] == 1) {
      // Because its specific value does not affect the final result, for
      // simplicity, it is set to 1 here.
      strides[i] = c1;
      continue;
    }
    int64_t inferredStride = axisInfo->getStrideValue(i);
    if (inferredStride != triton::AxisInfoExt::kStrideValueInitValue) {
      strides[i] = rewriter.create<arith::ConstantIndexOp>(loc, inferredStride);
      continue;
    }
    SmallVector<Value> indices = startIndices;
    indices[i] = c1;

    Value stride = rewriter.create<arith::SubIOp>(
        loc, rewriter.create<tensor::ExtractOp>(loc, offset, indices),
        scalarOffset);
    stride = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                 stride);
    strides[i] = stride;
  }

  if (maskTracker) {
    scalarOffset = rewriter.create<tensor::ExtractOp>(
        loc, offset,
        vector::getAsValues(rewriter, loc, maskTracker->getStarts()));
  }

  scalarOffset = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), scalarOffset);
  return {scalarOffset, strides};
}

// Abbreviation for PtrInfo.
using PtrInfo = TritonPtrLoadStoreOpConversionBase::PtrInfo;

FailureOr<PtrInfo> TritonPtrContiguousConversionBase::getPtrInfo(
    Location loc, Value ptr, Value mask, RankedTensorType tensorType,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode,
    bool allowMaskTrackerFailureIgnore) const {
  PtrInfo ret;
  const auto *axisInfo = getAxisInfo(ptr);
  ret.offsets =
      SmallVector<OpFoldResult>(tensorType.getRank(), rewriter.getIndexAttr(0));
  if (!axisInfo)
    return failure();

  // Analyze the mask operand to determine at runtime the size of the data we
  // move.
  MaskTracker maskTracker;
  if (mask) {
    maskTracker.parse(mask, loc, rewriter);
    // The ptrInfo analysis might have failed due to the mask tracker parse.
    // We may have chance to get the ptrInfo analysis result if allow to ignore
    // mask tracker parse failure.
    if (maskTracker.hasFailedDim()) {
      if (allowMaskTrackerFailureIgnore) {
        ret.isMaskTrackerFailed = true;
      } else {
        return failure();
      }
    } else {
      ret.offsets = llvm::to_vector(maskTracker.getStarts());
    }
  }

  ret.dimInfos = getDimInfos(axisInfo, tensorType.getShape());
  if (!isBlockPtr(ret.dimInfos))
    return failure();

  PointerMetaInfoTracker ptrInfoTracker;
  if (failed(ptrInfoTracker.parse(ptr, loc, rewriter)))
    return failure();
  ret.sizes = getActualSizes(
      loc, tensorType.getShape(),
      mask ? (ret.isMaskTrackerFailed ? std::nullopt
                                      : std::optional<MaskTracker>(maskTracker))
           : std::nullopt,
      rewriter);
  auto &&[offset, strides] = getOffsetAndStrides(
      loc, tensorType.getShape(), ptrInfoTracker.getOffset(),
      mask ? (ret.isMaskTrackerFailed ? std::nullopt
                                      : std::optional<MaskTracker>(maskTracker))
           : std::nullopt,
      axisInfo, rewriter);

  ret.permutations = getPermutations(axisInfo, tensorType.getShape());
  ret.memref =
      getMemRef(rewriter.getRemappedValue(ptrInfoTracker.getBase()),
                getAsOpFoldResult(offset), ret.sizes,
                getAsOpFoldResult(strides), ret.permutations, ret.dimInfos,
                tensorType.getElementType(), rewriter, cacheMode);
  ret.offsets = getMaskedOffsets(
      tensorType.getRank(),
      mask ? std::optional<MaskTracker>(maskTracker) : std::nullopt, rewriter);
  return ret;
}

//===----------------------------------------------------------------------===//
// TritonPtrScalarConversionBase
//===----------------------------------------------------------------------===//
Value TritonPtrScalarConversionBase::getMemRef(
    Location loc, Value ptr, Type elementType,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode) const {
  OpFoldResult c1 = rewriter.getIndexAttr(1);
  OpFoldResult c0 = rewriter.getIndexAttr(0);
  return TritonPtrConversionBase::getMemRef(rewriter.getRemappedValue(ptr), c0,
                                            {c1}, {c1}, {}, {}, elementType,
                                            rewriter, cacheMode);
}

//===----------------------------------------------------------------------===//
// TritonPtrScatterConversionBase
//===----------------------------------------------------------------------===//
Value TritonPtrScatterConversionBase::getDynamicMemRef(
    Location loc, Value ptr, RankedTensorType tensorType,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode) const {
  OpFoldResult c1 = rewriter.getIndexAttr(1);
  // Using the maximum size to represent the pointer size of unknown shape.
  OpFoldResult size =
      rewriter.getIndexAttr(std::numeric_limits<int64_t>().max());
  return getMemRef(rewriter.getRemappedValue(ptr), nullptr, {size}, {c1}, {},
                   {}, tensorType.getElementType(), rewriter, cacheMode);
}

//===----------------------------------------------------------------------===//
// TritonTensorPtrLoadStoreOpConversionBase
//===----------------------------------------------------------------------===//
SmallVector<OpFoldResult>
TritonTensorPtrLoadStoreOpConversionBase::getActualSizes(
    Location loc, std::optional<ArrayRef<int>> boundaryCheck,
    ArrayRef<int64_t> tensorShape, const TensorPointerMetaInfoTracker &tracker,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<OpFoldResult> blockSizes = llvm::to_vector<4>(
      llvm::map_range(tensorShape, [&rewriter](int64_t dim) -> OpFoldResult {
        return rewriter.getIndexAttr(dim);
      }));
  if (boundaryCheck) {
    for (auto i : boundaryCheck.value()) {
      OpFoldResult remainSize = subOFRs(tracker.getSizes()[i],
                                        tracker.getOffsets()[i], loc, rewriter);
      blockSizes[i] = minOFRs(remainSize, blockSizes[i], loc, rewriter);
    }
  }
  return blockSizes;
}

SmallVector<DimInfo> TritonTensorPtrLoadStoreOpConversionBase::getDimInfos(
    ArrayRef<OpFoldResult> strides, ArrayRef<int64_t> tensorShape) const {
  SmallVector<DimInfo> dimInfos;
  dimInfos.reserve(strides.size());
  for (auto [stride, size] : llvm::zip(strides, tensorShape)) {
    auto strideAsInt = getConstantIntValue(stride);
    if (strideAsInt.has_value() && strideAsInt.value() == 0) {
      dimInfos.push_back({1, size, DimInfo::Kind::BROADCAST});
      continue;
    }

    dimInfos.push_back({size, 1, DimInfo::Kind::CONTIG});
  }
  return dimInfos;
}
