//===- TritonPointerConversion.h - pointer conversion -----------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONPOINTERCONVERSION_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONPOINTERCONVERSION_H
#include <assert.h>
#include <optional>
#include <stdint.h>
#include <utility>

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class ConversionPatternRewriter;
class DataFlowSolver;
class Operation;

namespace triton {
class AxisInfoExt;
class TensorPointerMetaInfoTracker;
class MaskTracker;

/// Flatten value to match the definition of gather and scatter, if
/// appendUnitDim is set, append a shape of size one in the last.
///
/// Example 1:
/// ```mlir
///   %0 = foo ... : tensor<mxnxkxf32>
/// ```
/// is flatten to:
/// ```mlir
///   %0 = foo ... : tensor<mxnxkxf32>
///   %1 = tensor.collapse_shape %0 [[0, 1, 2]] :
///       tensor<mxnxkxf32> to tensor<sxf32> // s = m * n * k
///   %2 = tensor.expand_shape %1 [[0, 1]] : tensor<sxf32> to tensor<sx1xf32>
/// ```
///
/// Example 2:
/// ```mlir
///   %0 = foo ... : tensor<mxf32>
/// ```
/// is flatten to:
/// ```mlir
///   %0 = foo ... : tensor<mxf32>
///   %1 = tensor.expand_shape %0 [[0, 1]] : tensor<mxf32> to tensor<mx1xf32>
/// ```
///
/// Example 3:
/// ```mlir
///   %0 = foo ... : tensor<f32>
/// ```
/// is flatten to:
/// ```mlir
///   %0 = foo ... : tensor<f32>
///   %1 = tensor.expand_shape %0 [[]] : tensor<f32> to tensor<1x1xf32>
/// ```
Value flattenValueToMatchGatherScatter(ConversionPatternRewriter &rewriter,
                                       Value value, bool appendUnitDim = true);

/// Reshape the result of gather/scatter op to match the result type. Assume
/// the rank of value is srcRank, and the rank to reshape is dstRank. Then, the
/// first `srcRank - 1` dim size of value and resultTy must be same.
///
/// Example 1:
/// ```mlir
///   %0 = gather ... : tensor<sx1xf32>
///   resultTy: tensor<mxnxkxf32>
/// ```
/// is reshape to:
/// ```mlir
///   %0 = gather ... : tensor<sx1xf32>
///   %1 = tensor.collapse_shape %0 [[0, 1]] :
///       tensor<sx1xf32> to tensor<sxf32>
///   %2 = tensor.expand_shape %1 [[0, 1, 2]] : tensor<sxf32> to
///   tensor<mxnxkxf32>
/// ```
///
/// Example 2:
/// ```mlir
///   %0 = gather ... : tensor<mx1xf32>
///   resultTy: tensor<mxf32>
/// ```
/// is reshape to:
/// ```mlir
///   %0 = gather ... : tensor<mx1xf32>
///   %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<mx1xf32> to tensor<mxf32>
/// ```
///
/// Example 3:
/// ```mlir
///   %0 = gather ... : tensor<1x1xf32>
///   resultTy: tensor<f32>
/// ```
/// is reshape to:
/// ```mlir
///   %0 = gather ... : tensor<1x1xf32>
///   %1 = tensor.collapse_shape %0 [[]] : tensor<1x1xf32> to tensor<f32>
/// ```
Value reshapeGatherScatterValueTo(Value value, RankedTensorType resultTy,
                                  ConversionPatternRewriter &rewriter);

/// Select true value or false value based on a mask value.
Value selectByMask(Location loc, Value mask, Value trueVal, Value falseVal,
                   ConversionPatternRewriter &rewriter);

/// This class represents additional information bound to dimensions.
/// It contains two members, one is 'contigSize' and the other is
/// 'broadcastSize', which represent the length of continuity or broadcast
/// respectively. The product of these two values always equals
/// the length of that axis.
class DimInfo {
public:
  /// This enum represents the behavior of the innermost dimension.
  enum Kind : int32_t {
    /// It indicates that the innermost dimension is contiguous, similar to a
    /// sequence like `1234..k1234..k1234..k1234..k`. This means we first
    /// extract a continuous sequence of length k, which is `1234..k`, and then
    /// broadcast it into a 4xk sequence. Here, the innermost dimension is
    /// contiguous with size k.
    CONTIG = 0,
    /// It indicates that the innermost dimension is broadcast, similar to a
    /// sequence like `1111222233334444...kkkk`. This means we first
    /// extract a continuous sequence of length k, which is `1234..k`, and then
    /// broadcast it into a kx4 sequence. Here, the innermost dimension is
    /// broadcast with size 4.
    BROADCAST = 1,
    /// This dimension does not have any continuity or broadcast information,
    /// indicating a lowering to the case of gather/scatter.
    OTHERS = -1
  };
  DimInfo(int64_t contig, int64_t broadcast, Kind kind)
      : contigSize{contig}, broadcastSize{broadcast}, kind{kind} {
    assert(kind != Kind::OTHERS);
    dimSize = contigSize * broadcastSize;
  }

  DimInfo(int64_t dimSize) : dimSize{dimSize}, kind{Kind::OTHERS} {}

  Kind getKind() const { return kind; }
  int64_t getContigSize() const { return contigSize; }
  int64_t getBroadcastSize() const { return broadcastSize; }
  int64_t getDimSize() const { return dimSize; }
  bool isBroadcastDim() const { return getContigSize() == 1; }

private:
  int64_t contigSize = -1;
  int64_t broadcastSize = -1;
  int64_t dimSize = -1;
  Kind kind;
};

class TritonPtrConversionBase {
protected:
  /// Retrieve the corresponding memref based on baseptr, offsets, strides, and
  /// sizes, where permutations indicate that a transpose operation is required
  /// before obtaining the memref. As this pertains to memref, it's sufficient
  /// to transform the offsets/strides/sizes. Subsequently, use
  /// transformValueWithTransposeAndDimInfo to insert linalg.transpose,
  /// converting the corresponding tensor back. Likewise, process the dimInfos.
  Value getMemRef(Value basePtr, ArrayRef<OpFoldResult> offsets,
                  ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                  ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
                  Type elementType, ConversionPatternRewriter &rewriter,
                  StringAttr cacheMode = nullptr) const;

  /// Retrieve the corresponding memref based on baseptr, offset, strides and
  /// sizes.
  Value getMemRef(Value basePtr, OpFoldResult offset,
                  ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                  ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
                  Type elementType, ConversionPatternRewriter &rewriter,
                  StringAttr cacheMode = nullptr) const;

public:
  /// Based on the permutations and dimInfos information, insert
  /// linalg.transpose and linalg.broadcast operators to transform the result
  /// into the corresponding value.
  Value transformResultWithTransposeAndDimInfo(
      Value value, ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
      ArrayRef<OpFoldResult> actualSizes,
      ConversionPatternRewriter &rewriter) const;

  /// Based on the permutations and dimInfos information, insert
  /// linalg.transpose and tensor.extract_slice operators to transform the input
  /// into the corresponding value.
  Value transformInputWithTransposeAndDimInfo(
      Value value, ArrayRef<int64_t> permutations, ArrayRef<DimInfo> dimInfos,
      ArrayRef<OpFoldResult> offsets,
      ConversionPatternRewriter &rewriter) const;
};

/// Transform the values according to the rules based on the permutations.
/// For example, given values of size {a, b, c, d}, and if permutations are
/// {0,2,1,3}, the transformation results in {a, c, b, d}.
template <typename T>
inline SmallVector<T> getValuesByPerms(ArrayRef<T> values,
                                       ArrayRef<int64_t> perms) {
  if (perms.empty())
    return SmallVector<T>{values};
  assert(isPermutationVector(perms) && perms.size() == values.size());
  SmallVector<T> ret;
  ret.reserve(values.size());
  for (const auto &value : llvm::enumerate(values)) {
    ret.push_back(values[perms[value.index()]]);
  }
  return ret;
}

/// Transform the values according to the rules based on the dimInfos
/// information. If the dimInfo.contigSize equals to 1 for a particular
/// dimension, in order to match the definition of the broadcast operator, this
/// dimension needs to be removed, while keeping the others unchanged.
template <typename T>
static SmallVector<T> removeBroadcastDimByDimInfo(ArrayRef<T> values,
                                                  ArrayRef<DimInfo> dimInfos) {
  if (dimInfos.empty())
    return SmallVector<T>{values};
  SmallVector<T> ret;
  ret.reserve(values.size());
  for (auto &&[value, dimInfo] : llvm::zip(values, dimInfos)) {
    if (!dimInfo.isBroadcastDim()) {
      ret.push_back(value);
    }
  }
  return ret;
}

/// Based on the permutations and dimInfos information, obtain the shape
/// corresponding to the earlier getMemRef function, noting that the shape
/// referred to here is the padded static shape.
template <typename T>
SmallVector<T> permutateAndRemoveBroadcastDims(ArrayRef<T> values,
                                               ArrayRef<int64_t> permutations,
                                               ArrayRef<DimInfo> dimInfos) {
  return removeBroadcastDimByDimInfo<T>(
      getValuesByPerms<T>(values, permutations),
      getValuesByPerms<DimInfo>(dimInfos, permutations));
}

class TritonPtrLoadStoreOpConversionBase : public TritonPtrConversionBase {
public:
  struct PtrInfo {
    Value memref;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<DimInfo> dimInfos;
    SmallVector<int64_t> permutations;
    bool isMaskTrackerFailed = false;
  };

protected:
  TritonPtrLoadStoreOpConversionBase(mlir::DataFlowSolver &solver)
      : solver{solver} {}

  SmallVector<DimInfo> getDimInfos(const triton::AxisInfoExt *axisInfo,
                                   ArrayRef<int64_t> tensorShape) const;
  const triton::AxisInfoExt *getAxisInfo(Value ptr) const;
  SmallVector<int64_t> getPermutations(const triton::AxisInfoExt *axisInfo,
                                       ArrayRef<int64_t> tensorShape) const;

  /// Get the actual size of each dim needed to be load.
  SmallVector<OpFoldResult>
  getActualSizes(Location loc, ArrayRef<int64_t> tensorShape,
                 std::optional<triton::MaskTracker> maskTracker,
                 ConversionPatternRewriter &rewriter) const;

  bool isBlockPtr(ArrayRef<DimInfo> dimInfos) const;

  SmallVector<OpFoldResult>
  getMaskedOffsets(int64_t rank, std::optional<triton::MaskTracker> maskTracker,
                   ConversionPatternRewriter &rewriter) const;

private:
  mlir::DataFlowSolver &solver;
};

class TritonPtrContiguousConversionBase
    : public TritonPtrLoadStoreOpConversionBase {
  using TritonPtrLoadStoreOpConversionBase::TritonPtrLoadStoreOpConversionBase;

protected:
  std::pair<Value, SmallVector<Value>>
  getOffsetAndStrides(Location loc, ArrayRef<int64_t> tensorShape, Value offset,
                      std::optional<triton::MaskTracker> maskState,
                      const triton::AxisInfoExt *axisInfo,
                      ConversionPatternRewriter &rewriter) const;

  FailureOr<PtrInfo>
  getPtrInfo(Location loc, Value ptr, Value mask, RankedTensorType tensorType,
             ConversionPatternRewriter &rewriter,
             StringAttr cacheMode = nullptr,
             bool allowMaskTrackerFailureIgnore = false) const;
};

class TritonPtrScalarConversionBase
    : public TritonPtrLoadStoreOpConversionBase {
  using TritonPtrLoadStoreOpConversionBase::TritonPtrLoadStoreOpConversionBase;

protected:
  Value getMemRef(Location loc, Value ptr, Type elementType,
                  ConversionPatternRewriter &rewriter,
                  StringAttr cacheMode = nullptr) const;
};

class TritonPtrScatterConversionBase
    : public TritonPtrLoadStoreOpConversionBase {
  using TritonPtrLoadStoreOpConversionBase::TritonPtrLoadStoreOpConversionBase;

protected:
  /// Get the dynamic memref, it returns as follow:
  /// ```mlir
  ///   %tensor = aux.view %ptr : memref<9223372036854775807xf32>
  /// ```
  Value getDynamicMemRef(Location loc, Value ptr, RankedTensorType tensorType,
                         ConversionPatternRewriter &rewriter,
                         StringAttr cacheMode = nullptr) const;
};

class TritonTensorPtrLoadStoreOpConversionBase
    : public TritonPtrConversionBase {
protected:
  /// Get the actual size of each dim needed to be load, if boundaryCheck is
  /// true, return min(tensorShape[dim], dimSize[dim] - offset[dim]).
  SmallVector<OpFoldResult>
  getActualSizes(Location loc, std::optional<ArrayRef<int>> boundaryCheck,
                 ArrayRef<int64_t> tensorShape,
                 const TensorPointerMetaInfoTracker &tracker,
                 ConversionPatternRewriter &rewriter) const;

  SmallVector<DimInfo> getDimInfos(ArrayRef<OpFoldResult> strides,
                                   ArrayRef<int64_t> tensorShape) const;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONPOINTERCONVERSION_H
