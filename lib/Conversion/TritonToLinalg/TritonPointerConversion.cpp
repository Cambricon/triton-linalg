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

#include "triton-linalg/Conversion/TritonToLinalg/TritonPointerConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/Triton/Analysis/AxisInfoAnalysis.h"
#include "triton-linalg/Dialect/Triton/Interfaces/InferAxisInfoInterface.h"
#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Dialect/Triton/Utils/OpFoldResultUtils.h"
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
  auto initDims = getDims(rewriter, loc, falseVal);
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

/// Check if any broadcast dimensions need mask, i.e., if the masked size
/// is smaller than original size.
static bool
shouldApplyMaskForAnyBroadcastDim(ArrayRef<DimInfo> dimInfos,
                                  ArrayRef<OpFoldResult> maskedSizes) {
  // Identify broadcast dimensions.
  SmallVector<int64_t> broadcastDimensions;
  for (const auto &[index, dimInfo] : llvm::enumerate(dimInfos)) {
    if (dimInfo.isBroadcastDim())
      broadcastDimensions.push_back(index);
  }

  // Check whether masks are needed for any broadcast dimension.
  return llvm::any_of(broadcastDimensions,
                      [maskedSizes, dimInfos](int64_t dim) {
                        auto size = getConstantIntValue(maskedSizes[dim]);
                        return !size || *size != dimInfos[dim].getDimSize();
                      });
}

/// Extract slice `value` based on offsets/sizes/strides, and then drop
/// the last `dimsToDrop` dims.
///
/// Example 1:
/// Parameters:
/// - offsets = [0, 1, 0, 0, 1]
/// - sizes   = [1, 3, 2, 1, 3]
/// - strides = [1, 1, 1, 1, 1]
/// - dimsToDrop = 2
/// ```mlir
/// %0 = tensor.empty() : tensor<1x4x2x1x4xf32>
/// %1 = tensor.extract_slice %0[0, 1, 0, 0, 1][1, 3, 2, 1, 1][1, 1, 1, 1, 1]
///      : tensor<1x4x2x1x4xf32> to tensor<1x3x2xf32>
/// ```
static Value extractSliceAndDropLastNdims(Location loc, Value value,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes,
                                          ArrayRef<OpFoldResult> strides,
                                          int64_t dimsToDrop,
                                          ConversionPatternRewriter &rewriter) {
  auto srcType = value.getType().cast<RankedTensorType>();
  int64_t dstRank = sizes.size() - dimsToDrop;
  // Reset the size of last n dims to 1.
  SmallVector<OpFoldResult> resetSizes(sizes);
  std::for_each(resetSizes.begin() + dstRank, resetSizes.end(),
                [&rewriter](auto &ofr) { ofr = rewriter.getIndexAttr(1); });
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  staticSizes.resize(dstRank);
  auto dstType = RankedTensorType::get(staticSizes, srcType.getElementType());

  return rewriter
      .create<tensor::ExtractSliceOp>(loc, dstType, value, offsets, resetSizes,
                                      strides)
      .getResult();
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

const AxisInfo *
TritonPtrLoadStoreOpConversionBase::getAxisInfo(Value ptr) const {
  auto *lattice = solver.lookupState<AxisInfoLattice>(ptr);
  if (!lattice)
    return nullptr;
  return &lattice->getValue();
}

SmallVector<DimInfo> TritonPtrLoadStoreOpConversionBase::getDimInfos(
    const AxisInfo *axisInfo, ArrayRef<int64_t> tensorShape) const {
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
    const AxisInfo *axisInfo, ArrayRef<int64_t> tensorShape) const {
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
    std::optional<MaskTracker> maskTracker, const AxisInfo *axisInfo,
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
    if (inferredStride != AxisInfo::kStrideValueInitValue) {
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
int64_t TritonPtrScatterConversionBase::getWindowRank(
    ArrayRef<int64_t> shape, ArrayRef<int64_t> perms,
    std::optional<MaskTracker> tracker, const AxisInfo &axisInfo) const {
  auto rank = shape.size();
  int64_t windowRank = 0;
  int64_t stride = 1;
  for (size_t i = 0; i < rank; ++i) {
    auto dim = perms[rank - 1 - i];
    // If fail to analyze mask tracker, stop searching.
    if (tracker && !tracker->getStarts()[dim])
      break;

    // Check whether current dimension is broadcastable.
    if (axisInfo.isConstantDim(shape, dim))
      continue;

    // Check whether current dimension is contiguous.
    if (!axisInfo.isStrideDim(shape, dim) ||
        axisInfo.getStrideValue(dim) != stride)
      break;

    ++windowRank;

    // If only use part of current dim, stop searching.
    if (tracker) {
      auto size = getConstantIntValue(tracker->getSizes()[dim]);
      if (!size || size != shape[dim])
        break;
    }

    // Update stride value for previous dim.
    stride *= shape[dim];
  }

  return windowRank;
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

Value TritonPtrScatterConversionBase::getIndiceOperandForScatteredOp(
    Location loc, Value originIndice, int64_t windowRank,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
    ConversionPatternRewriter &rewriter) const {
  auto zeroAttr = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> defaultOffsets(dimInfos.size(), zeroAttr);
  auto indice = transformInputWithTransposeAndDimInfo(
      originIndice, permutations, dimInfos, defaultOffsets, rewriter);

  auto indiceType = indice.getType().dyn_cast<RankedTensorType>();
  int64_t rank = indiceType.getRank();
  unsigned batch = rank - windowRank;

  // Set extract slice sizes info by resetting the dim size of window to 1.
  auto indiceSizes = permutateAndRemoveBroadcastDims<OpFoldResult>(
      sizes, permutations, dimInfos);
  auto oneAttr = rewriter.getIndexAttr(1);
  std::fill(indiceSizes.begin() + batch, indiceSizes.end(),
            OpFoldResult(oneAttr));

  // Set extract slice offsets info.
  // 1. Set offsets of batch dims and the first window dim to the start
  // of mask state.
  // 2. Set offsets of other window dims to 0.
  auto transposedOffsets = permutateAndRemoveBroadcastDims<OpFoldResult>(
      offsets, permutations, dimInfos);
  SmallVector<OpFoldResult> indiceOffsets(rank, zeroAttr);
  std::copy(transposedOffsets.begin(), transposedOffsets.begin() + batch,
            indiceOffsets.begin());
  // Add mask start of the first window dim.
  if (windowRank > 0)
    indiceOffsets[batch] = transposedOffsets[batch];

  // Set extract slice strides to 1.
  SmallVector<OpFoldResult> indiceStrides(rank, oneAttr);
  indice = extractSliceAndDropLastNdims(loc, indice, indiceOffsets, indiceSizes,
                                        indiceStrides, rank - batch, rewriter);
  // Add a unit dim to the last.
  indice = appendUnitDim(rewriter, loc, indice);

  return indice;
}

Value TritonPtrScatterConversionBase::getMaskOperandForScatteredOp(
    Location loc, Value originMask, int64_t windowRank,
    ArrayRef<OpFoldResult> sizes, ArrayRef<int64_t> permutations,
    ArrayRef<DimInfo> dimInfos, std::optional<MaskTracker> maskTracker,
    ConversionPatternRewriter &rewriter) const {
  if (!maskTracker)
    return Value();

  // If the mask tracker for a broadcast dim fails, perform reduction with
  // arith.ori operation to determine whether the specific indice is masked.
  SmallVector<int64_t> reductionDims;
  for (const auto &[index, dimInfo] : llvm::enumerate(dimInfos)) {
    if (dimInfo.isBroadcastDim() &&
        !static_cast<bool>(maskTracker->getSizes()[index]))
      reductionDims.push_back(index);
  }
  Value mask = originMask;
  if (!reductionDims.empty()) {
    auto maskTy = originMask.getType().cast<ShapedType>();
    assert((!ShapedType::isDynamicShape(maskTy.getShape())) &&
           "value shape should be static");
    SmallVector<int64_t> shape(maskTy.getShape());
    SmallVector<int64_t> initShape;
    for (size_t i = 0, j = 0; i < shape.size(); ++i) {
      if (j >= reductionDims.size() || i != reductionDims[j])
        initShape.push_back(shape[i]);
      else
        ++j;
    }
    Value init = rewriter.create<tensor::EmptyOp>(loc, initShape,
                                                  maskTy.getElementType());
    auto identity = getIdentityValueAttr(
        arith::AtomicRMWKind::ori, maskTy.getElementType(), rewriter, loc);
    Value constantOp = rewriter.create<arith::ConstantOp>(loc, identity);
    Value reductionInit = rewriter
                              .create<linalg::FillOp>(
                                  loc, ValueRange{constantOp}, ValueRange{init})
                              .result();
    Value reductionMask =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{originMask}, ValueRange{reductionInit},
                reductionDims,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  Value res = b.create<arith::OrIOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, res);
                })
            .getResult(0);
    // Broadcast the reduced mask to the original shape.
    Value maskInit =
        rewriter.create<tensor::EmptyOp>(loc, shape, maskTy.getElementType());
    mask = rewriter
               .create<linalg::BroadcastOp>(loc, reductionMask, maskInit,
                                            reductionDims)
               ->getResult(0);
  }

  auto zeroAttr = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> starts(maskTracker->getStarts());
  SmallVector<OpFoldResult> transposedStarts =
      permutateAndRemoveBroadcastDims<OpFoldResult>(starts, permutations,
                                                    dimInfos);
  std::for_each(starts.begin(), starts.end(), [zeroAttr](auto &ofr) {
    ofr = ofr ? ofr : OpFoldResult(zeroAttr);
  });
  int64_t rank = transposedStarts.size();
  int64_t batch = rank - windowRank;
  // Cases which need mask in gather/scatter operations:
  // 1. For cases with batch = 0.
  // 2. For cases with batch > 0, and there exists batch dims whose mask is
  // failed to analyze.
  bool needMask =
      batch == 0 ||
      std::any_of(transposedStarts.begin(), transposedStarts.begin() + batch,
                  [](auto ofr) { return !static_cast<bool>(ofr); });
  if (!needMask)
    return Value();

  // Only assign the starts of mask tracker for broadcast dimensions.
  for (const auto &dimInfo : llvm::enumerate(dimInfos))
    if (!dimInfo.value().isBroadcastDim())
      starts[dimInfo.index()] = zeroAttr;
  mask = transformInputWithTransposeAndDimInfo(mask, permutations, dimInfos,
                                               starts, rewriter);
  // Drop the last windowRank dims.
  auto oneAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<OpFoldResult> transposedSizes =
      permutateAndRemoveBroadcastDims<OpFoldResult>(sizes, permutations,
                                                    dimInfos);
  std::for_each(
      transposedStarts.begin(), transposedStarts.end(),
      [zeroAttr](auto &ofr) { ofr = ofr ? ofr : OpFoldResult(zeroAttr); });
  return extractSliceAndDropLastNdims(loc, mask, transposedStarts,
                                      transposedSizes, strides, windowRank,
                                      rewriter);
}

bool TritonPtrScatterConversionBase::checkIfConsiderMaskForWindowAnalysis(
    Operation *op, ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
    const MaskTracker &tracker) const {
  if (auto storeOp = dyn_cast<triton::StoreOp>(op))
    return static_cast<bool>(storeOp.getMask());

  auto loadOp = dyn_cast<triton::LoadOp>(op);
  assert(loadOp && "unsupport operation type.");
  // Retrieve the memory model based on module attribute.
  auto module = op->getParentOfType<ModuleOp>();
  if (!isLinearMemory(module))
    return static_cast<bool>(loadOp.getMask());

  if (!loadOp.getOther())
    return false;

  // If all dims are analyzed successfully, just igonore mask.
  if (llvm::all_of(tracker.getStarts(),
                   [](auto ofr) { return static_cast<bool>(ofr); }))
    return false;

  auto starts = permutateAndRemoveBroadcastDims<OpFoldResult>(
      tracker.getStarts(), permutations, dimInfos);
  if (starts.empty())
    return false;

  // There is a tradeoff between two strategies when other exists:
  // a. Contiguously load and select by mask; b. Masked load.
  // Currently, we force to contiguously load only when mask of the last
  // dimension is failed to analyze.
  return static_cast<bool>(starts.back());
}

// Abbreviation for ScatteredInfo.
using ScatteredInfo = TritonPtrScatterConversionBase::ScatteredInfo;

FailureOr<std::pair<PtrInfo, ScatteredInfo>>
TritonPtrScatterConversionBase::getPtrAndScatterInfo(
    Location loc, Value ptr, Value mask, MaskTracker &maskTracker,
    RankedTensorType tensorType, Operation *op,
    ConversionPatternRewriter &rewriter, StringAttr cacheMode) const {
  PtrInfo ptrInfo;
  auto tensorShape = tensorType.getShape();

  const auto *axisInfo = getAxisInfo(ptr);
  // Set the pessimistic axis info if analysis fails.
  AxisInfo pessmisticInfo = AxisInfo::getPessimisticValueState(ptr);
  if (!axisInfo)
    axisInfo = &pessmisticInfo;

  // Analyze the dim order based on axisInfo.
  ptrInfo.permutations = getPermutations(axisInfo, tensorShape);

  // Analyze dim info.
  ptrInfo.dimInfos = getDimInfos(axisInfo, tensorShape);

  int64_t windowRank;
  std::optional<MaskTracker> windowAnalysisMask = std::nullopt;
  if (checkIfConsiderMaskForWindowAnalysis(op, ptrInfo.permutations,
                                           ptrInfo.dimInfos, maskTracker))
    windowAnalysisMask = maskTracker;
  windowRank = getWindowRank(tensorShape, ptrInfo.permutations,
                             windowAnalysisMask, *axisInfo);

  int64_t rank = tensorType.getRank();
  int64_t nonConstantDimNum = 0;
  for (auto index : llvm::seq<int64_t>(0, rank))
    if (!axisInfo->isConstantDim(tensorShape, index))
      ++nonConstantDimNum;

  if (mask) {
    for (int64_t i = 0, nonConstantIndex = 0; i < rank; ++i) {
      auto dim = ptrInfo.permutations[rank - 1 - i];
      if (axisInfo->isConstantDim(tensorShape, dim))
        continue;

      // Only handle non-highest batch dimensions.
      if (nonConstantIndex >= windowRank &&
          nonConstantIndex < nonConstantDimNum - 1)
        maskTracker.setDimensionStatus(dim, nullptr, nullptr, nullptr);

      ++nonConstantIndex;
    }
  }

  PointerMetaInfoTracker tracker;
  if (failed(tracker.parse(ptr, loc, rewriter)))
    return failure();

  ptrInfo.sizes = getActualSizes(
      loc, tensorType.getShape(),
      mask ? std::optional<MaskTracker>(maskTracker) : std::nullopt, rewriter);

  ptrInfo.memref =
      getDynamicMemRef(loc, tracker.getBase(), tensorType, rewriter, cacheMode);

  auto indice = tracker.getOffset();

  ptrInfo.offsets = getMaskedOffsets(
      tensorType.getRank(),
      mask ? std::optional<MaskTracker>(maskTracker) : std::nullopt, rewriter);

  ScatteredInfo scatteredInfo;
  scatteredInfo.windowRank = windowRank;

  scatteredInfo.indice = getIndiceOperandForScatteredOp(
      loc, indice, scatteredInfo.windowRank, ptrInfo.offsets, ptrInfo.sizes,
      ptrInfo.permutations, ptrInfo.dimInfos, rewriter);

  scatteredInfo.mask = getMaskOperandForScatteredOp(
      loc, mask, scatteredInfo.windowRank, ptrInfo.sizes, ptrInfo.permutations,
      ptrInfo.dimInfos,
      mask ? std::optional<MaskTracker>(maskTracker) : std::nullopt, rewriter);

  return std::make_pair(ptrInfo, scatteredInfo);
}

/// Post process the gather result and get the final result.
/// 1. Transforms gathered result and expand broadcast dimensions;
/// 2. Selects gathered result or the `other` field by mask.
Value TritonPtrScatterConversionBase::postProcessGatheredResult(
    Location loc, Value result, Value mask, Value other,
    ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    RankedTensorType resultType, MaskTracker &maskTracker,
    ConversionPatternRewriter &rewriter) const {
  if (!mask)
    return transformResultWithTransposeAndDimInfo(result, permutations,
                                                  dimInfos, sizes, rewriter);

  // Check whether masks for all axes are parsed successfully.
  auto maskTrackerStatus =
      LogicalResult::success(!maskTracker.getSizes().empty() &&
                             llvm::all_of(maskTracker.getSizes(), [](auto ofr) {
                               return static_cast<bool>(ofr);
                             }));

  SmallVector<OpFoldResult> resultSizes = llvm::to_vector(llvm::map_range(
      resultType.getShape(), [&rewriter](int64_t size) -> OpFoldResult {
        return rewriter.getIndexAttr(size);
      }));

  // Handle special cases:
  // 1. When there is a dimension whose mask tracker fails.
  // 2. When the 'other' field is null.
  // In such cases, we perform the following steps:
  // a. Pad the gather result initially with empty value.
  // b. Transform the intermediate result using dimension information and
  // permutations.
  // c. Finally, select the padded result or 'other' based on the mask.(Do
  // nothing when other is null.)
  if (failed(maskTrackerStatus) || !other) {
    // Obtain transposed other type.
    SmallVector<int64_t> otherShape = permutateAndRemoveBroadcastDims(
        resultType.getShape(), permutations, dimInfos);
    auto otherType =
        RankedTensorType::get(otherShape, resultType.getElementType());
    // Obtain transposed offsets and sizes.
    auto transposedOffsets = permutateAndRemoveBroadcastDims<OpFoldResult>(
        offsets, permutations, dimInfos);
    auto transposedSizes = permutateAndRemoveBroadcastDims<OpFoldResult>(
        sizes, permutations, dimInfos);

    // Pad with empty value.
    result =
        getPadOrInsertOpWithOther(loc, Value(), otherType, result,
                                  transposedOffsets, transposedSizes, rewriter);

    // Transposed result back.
    result = transformResultWithTransposeAndDimInfo(
        result, permutations, dimInfos, resultSizes, rewriter);

    // Select the final result based on the mask.
    return selectByMask(loc, mask, result, other, rewriter);
  }

  // Handle a special case:
  // * All dimension masks have been successfully analyzed, and the sizes
  //   after masking are equal to the original sizes for broadcast dimensions.
  // In such cases, we perform the following steps:
  // a. Pad the gather result initially with the transformed `other`.
  // b. Transform the intermediate result using dimension information and
  // permutations.
  if (!shouldApplyMaskForAnyBroadcastDim(dimInfos, sizes)) {
    // Obtain transposed other type.
    SmallVector<int64_t> otherShape = permutateAndRemoveBroadcastDims(
        resultType.getShape(), permutations, dimInfos);
    auto otherType =
        RankedTensorType::get(otherShape, resultType.getElementType());
    // Obtain transposed offsets and sizes.
    auto transposedOffsets = permutateAndRemoveBroadcastDims<OpFoldResult>(
        offsets, permutations, dimInfos);
    auto transposedSizes = permutateAndRemoveBroadcastDims<OpFoldResult>(
        sizes, permutations, dimInfos);

    // Pad with transformed `other`.
    auto zeroAttr = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> defaultOffsets(dimInfos.size(), zeroAttr);
    auto transformedOther = transformInputWithTransposeAndDimInfo(
        other, permutations, dimInfos, defaultOffsets, rewriter);
    result =
        getPadOrInsertOpWithOther(loc, transformedOther, otherType, result,
                                  transposedOffsets, transposedSizes, rewriter);

    // Transpose result back.
    return transformResultWithTransposeAndDimInfo(
        result, permutations, dimInfos, resultSizes, rewriter);
  }

  // For cases that all dimension masks have been successfully analyzed, and
  // part of broadcast dimensions have masks. We perform the following steps:
  // a. Transform the gather result and broadcast `BROADCAST` dims to masked
  // size.
  // b. Pad the intermediate result with `other`.
  result = transformResultWithTransposeAndDimInfo(result, permutations,
                                                  dimInfos, sizes, rewriter);
  return getPadOrInsertOpWithOther(loc, other, resultType, result, offsets,
                                   sizes, rewriter);
}

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
