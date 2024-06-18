//===- LinalgExtOpTilingInterface.cpp -Ext TilingInterface impl--*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <memory>
#include <optional>
#include <stddef.h>
#include <stdint.h>
#include <tuple>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h" // IWYU pragma: keep
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/LinalgExt/Transforms/TilingInterfaceImpl.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h" // IWYU pragma: keep
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
using namespace mlir;
using namespace mlir::triton;
namespace mlir {
class MLIRContext;
} // namespace mlir

static Value getSimpliedSlice(OpBuilder &b, Location loc, Value source,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides) {
  auto simpliedOffsets = canonicalizeOpFoldResult(offsets);
  auto simpliedSizes = canonicalizeOpFoldResult(sizes);
  auto simpliedStrides = canonicalizeOpFoldResult(strides);
  return getSlice(b, loc, source, simpliedOffsets, simpliedSizes,
                  simpliedStrides);
}

/// Add a linalg.broadcast op and return its output.
static Value addBroadcast(OpBuilder &builder, Location loc, Value input,
                          ArrayRef<OpFoldResult> shapeOperands,
                          ArrayRef<int64_t> broadcastDim) {
  ShapedType shapeTy = input.getType().cast<ShapedType>();
  Value init = builder.create<tensor::EmptyOp>(loc, shapeOperands,
                                               shapeTy.getElementType());
  return builder.create<linalg::BroadcastOp>(loc, input, init, broadcastDim)
      ->getResult(0);
}

/// Add a linalg.map op and return its output.
using RegionFn = function_ref<void(OpBuilder &, Location, ValueRange)>;
static Value addMap(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                    RegionFn regionFn) {
  auto shapeOperands = getDims(builder, loc, lhs);
  ShapedType shapeTy = lhs.getType().cast<ShapedType>();
  Value init = builder.create<tensor::EmptyOp>(loc, shapeOperands,
                                               shapeTy.getElementType());
  return builder
      .create<linalg::MapOp>(loc, ValueRange{lhs, rhs}, init, regionFn)
      ->getResult(0);
}

/// Padded tiled indice to the paddedLength.
static Value padTiledIndice(OpBuilder &builder, Location loc, Value input,
                            int64_t paddedLength) {
  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type eleTy = inputTy.getElementType();
  Value zero = builder.create<arith::ConstantOp>(
      loc, eleTy, builder.getIntegerAttr(eleTy, 0));
  SmallVector<int64_t> dstShape = llvm::to_vector(inputTy.getShape());
  dstShape.back() = paddedLength;
  RankedTensorType dstTy = RankedTensorType::get(dstShape, eleTy);
  return tensor::createPadHighOp(dstTy, input, zero, false, loc, builder);
}

/// Construct memref.atomic_rmw/memref.generic_atomic_rmw based on
/// triton::linalg_ext::AtomicType.
static LogicalResult
createMemrefAtomicRMW(OpBuilder &b, Location loc, Value value, Value memref,
                      ValueRange indices, triton::linalg_ext::AtomicType type) {
  OpBuilder::InsertionGuard guard(b);
  arith::AtomicRMWKind kind;
  switch (type) {
  case triton::linalg_ext::AtomicType::andi:
    kind = arith::AtomicRMWKind::andi;
    break;
  case triton::linalg_ext::AtomicType::ori:
    kind = arith::AtomicRMWKind::ori;
    break;
  case triton::linalg_ext::AtomicType::addf:
    kind = arith::AtomicRMWKind::addf;
    break;
  case triton::linalg_ext::AtomicType::addi:
    kind = arith::AtomicRMWKind::addi;
    break;
  case triton::linalg_ext::AtomicType::maximumf:
    kind = arith::AtomicRMWKind::maximumf;
    break;
  case triton::linalg_ext::AtomicType::maxs:
    kind = arith::AtomicRMWKind::maxs;
    break;
  case triton::linalg_ext::AtomicType::maxu:
    kind = arith::AtomicRMWKind::maxu;
    break;
  case triton::linalg_ext::AtomicType::minimumf:
    kind = arith::AtomicRMWKind::minimumf;
    break;
  case triton::linalg_ext::AtomicType::mins:
    kind = arith::AtomicRMWKind::mins;
    break;
  case triton::linalg_ext::AtomicType::minu:
    kind = arith::AtomicRMWKind::minu;
    break;
  case triton::linalg_ext::AtomicType::xchg: {
    auto genericAtomicRMWOp =
        b.create<memref::GenericAtomicRMWOp>(loc, memref, indices);
    Block *blk = &genericAtomicRMWOp.body().front();
    b.setInsertionPointToStart(blk);
    b.create<memref::AtomicYieldOp>(loc, value);
    return success();
  }
  case triton::linalg_ext::AtomicType::xori: {
    auto genericAtomicRMWOp =
        b.create<memref::GenericAtomicRMWOp>(loc, memref, indices);
    Block *blk = &genericAtomicRMWOp.body().front();
    b.setInsertionPointToStart(blk);
    Value res = b.create<arith::XOrIOp>(loc, blk->getArgument(0), value);
    b.create<memref::AtomicYieldOp>(loc, res);
    return success();
  }
  default:
    llvm_unreachable("Invalid AtomicRMWType");
  }
  b.create<memref::AtomicRMWOp>(loc, value.getType(), kind, value, memref,
                                indices);
  return success();
}

/// Tile window and return sliced inputs:window, indices, mask.
static SmallVector<Value>
tileByWindowSlice(OpBuilder &b, Location loc, Value data, Value window,
                  Value indices, Value mask, SmallVector<int64_t> &dimensionMap,
                  ArrayRef<OpFoldResult> windowOffsets,
                  ArrayRef<OpFoldResult> windowSizes, int64_t batchNum) {
  auto zeroAttr = b.getI64IntegerAttr(0);
  auto oneAttr = b.getI64IntegerAttr(1);

  // Slice of init.
  auto windowRank = window.getType().cast<ShapedType>().getRank();
  SmallVector<OpFoldResult> windowStrides(windowRank, oneAttr);
  Value tiledWindow = getSimpliedSlice(b, loc, window, windowOffsets,
                                       windowSizes, windowStrides);
  assert(tiledWindow && "failed to get slice of window");
  // Slice of indices.
  auto indicesTy = indices.getType().cast<ShapedType>();
  auto indicesRank = indicesTy.getRank();
  SmallVector<OpFoldResult> indicesOffsets(indicesRank, zeroAttr);
  SmallVector<OpFoldResult> indicesSizes(indicesRank);
  for (auto i = 0; i < batchNum; i++) {
    indicesOffsets[i] = windowOffsets[i];
    indicesSizes[i] = windowSizes[i];
  }
  for (auto dim : llvm::seq<int64_t>(batchNum, indicesRank)) {
    indicesSizes[dim] = getDim(b, loc, indices, dim);
  }
  SmallVector<OpFoldResult> indicesStrides(indicesRank, oneAttr);
  Value tiledIndices = getSimpliedSlice(b, loc, indices, indicesOffsets,
                                        indicesSizes, indicesStrides);
  assert(tiledIndices && "failed to get slice of indices");
  // Slice of mask.
  Value tiledMask = mask;
  if (tiledMask) {
    SmallVector<OpFoldResult> maskOffsets(windowOffsets.begin(),
                                          windowOffsets.begin() + batchNum);
    SmallVector<OpFoldResult> maskSizes{windowSizes.begin(),
                                        windowSizes.begin() + batchNum};
    SmallVector<OpFoldResult> maskStrides(batchNum, oneAttr);
    tiledMask =
        getSimpliedSlice(b, loc, mask, maskOffsets, maskSizes, maskStrides);
    assert(tiledMask && "failed to get slice of mask");
  }
  // Update indice value using update tiling offset.
  auto dataRank = data.getType().cast<ShapedType>().getRank();
  ArrayRef<OpFoldResult> curOffsetArray(windowOffsets.begin() + batchNum,
                                        windowOffsets.end());
  bool hasNonZeroVal = llvm::any_of(curOffsetArray, [](OpFoldResult val) {
    auto valInt = getConstantIntValue(val);
    return !valInt.has_value() || valInt.value() != 0;
  });
  // If all non batch offset is 0, no need to recompute indices,
  // so we can use origin indice directly.
  if (hasNonZeroVal) {
    auto curOffsetVals =
        getValueOrCreateConstantIndexOp(b, loc, curOffsetArray);
    auto indiceEleTy = indicesTy.getElementType();
    for (auto &val : curOffsetVals) {
      val = b.create<arith::IndexCastOp>(loc, indiceEleTy, val);
    }
    Value curDataOffset = b.create<tensor::FromElementsOp>(loc, curOffsetVals);
    // Pad tiled indice.
    if (dataRank != indicesTy.getDimSize(1)) {
      tiledIndices = padTiledIndice(b, loc, tiledIndices, dataRank);
    }
    Value batchDataOffset =
        addBroadcast(b, loc, curDataOffset, getDims(b, loc, tiledIndices),
                     llvm::to_vector(llvm::seq<int64_t>(0, batchNum)));
    tiledIndices = addMap(b, loc, tiledIndices, batchDataOffset,
                          [](OpBuilder &b, Location loc, ValueRange args) {
                            Value innerResult =
                                b.create<arith::AddIOp>(loc, args[0], args[1]);
                            b.create<linalg::YieldOp>(loc, innerResult);
                          });
    // If tiled indice is padded, we need to pad dimension map to tiled indice
    // rank.
    SmallVector<int8_t, 8> record(dataRank, 0);
    for (auto val : dimensionMap)
      record[val] = 1;

    for (const auto &it : llvm::enumerate(record)) {
      if (it.value() == 0) {
        dimensionMap.push_back(it.index());
      }
    }
  }

  return {tiledWindow, tiledIndices, tiledMask};
}

namespace {
template <typename OpTy> struct LinalgExtOpTilingInterface {};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::ScatterOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::ScatterOp>,
          triton::linalg_ext::ScatterOp> {
  /// Return the destination operands.
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<mlir::DestinationStyleOpInterface>(op).getDpsInits();
  }

  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::ScatterOp concreteOp =
        cast<triton::linalg_ext::ScatterOp>(op);
    // Now, it is difficult to implement tiling init,
    // so we only tile update.
    SmallVector<utils::IteratorType> loops(concreteOp.getUpdateType().getRank(),
                                           utils::IteratorType::parallel);
    // If windows overlap directly, windows between
    // different batches are reduction iterator type.
    if (concreteOp.getOverlapWindow()) {
      auto batchNum = concreteOp.getBatchDimNum();
      for (int64_t i = 0; i < batchNum; i++)
        loops[i] = utils::IteratorType::reduction;
    }
    return loops;
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    triton::linalg_ext::ScatterOp concreteOp =
        cast<triton::linalg_ext::ScatterOp>(op);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, concreteOp.getUpdateType().getRank())) {
      OpFoldResult ub = getDim(b, loc, concreteOp.update(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  // Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
    // specified could lead to out of bounds accesses.
    triton::linalg_ext::ScatterOp concreteOp =
        cast<triton::linalg_ext::ScatterOp>(op);
    SmallVector<int64_t> dimensionMap =
        llvm::to_vector(concreteOp.getDimensionMap());
    auto loc = op->getLoc();
    Value init = concreteOp.getInit();
    auto batchNum = concreteOp.getBatchDimNum();
    // Tile window.
    auto windowIndicesAndMask = tileByWindowSlice(
        b, loc, init, concreteOp.update(), concreteOp.indice(),
        concreteOp.mask(), dimensionMap, offsets, sizes, batchNum);
    // Slice of the init.
    auto initRank = concreteOp.getInitType().getRank();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> initStrides(initRank, oneAttr);
    Value tiledInit;
    SmallVector<OpFoldResult> initOffsets, initSizes;
    initOffsets = SmallVector<OpFoldResult>(initRank, b.getIndexAttr(0));
    initSizes = getDims(b, loc, init);
    tiledInit =
        getSimpliedSlice(b, loc, init, initOffsets, initSizes, initStrides);
    assert(tiledInit && "failed to get slice of init");

    // Create tiled scatter.
    triton::linalg_ext::ScatterOp tiledScatterOp;
    if (windowIndicesAndMask[2]) {
      tiledScatterOp = b.create<triton::linalg_ext::ScatterOp>(
          loc,
          ValueRange({windowIndicesAndMask[0], windowIndicesAndMask[1],
                      windowIndicesAndMask[2]}),
          tiledInit, dimensionMap, concreteOp.getRangedData(),
          concreteOp.getOverlapWindow(), concreteOp.getSignedIndice());
    } else {
      tiledScatterOp = b.create<triton::linalg_ext::ScatterOp>(
          loc, ValueRange({windowIndicesAndMask[0], windowIndicesAndMask[1]}),
          tiledInit, dimensionMap, concreteOp.getRangedData(),
          concreteOp.getOverlapWindow(), concreteOp.getSignedIndice());
    }

    // Clean body region.
    auto &targetRegion = tiledScatterOp.getRegion();
    targetRegion.dropAllReferences();
    targetRegion.getBlocks().clear();

    // Clone body region from origin op.
    IRMapping map;
    concreteOp.getRegion().cloneInto(&targetRegion, map);
    return TilingResult{{tiledScatterOp},
                        SmallVector<Value>(tiledScatterOp->getResults())};
  }

  // Return the details of the output tile generated by the tiled
  // implementation.
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    triton::linalg_ext::ScatterOp concreteOp =
        cast<triton::linalg_ext::ScatterOp>(op);
    auto zeroAttr = b.getI64IntegerAttr(0);
    // Slice of the init.
    auto initRank = concreteOp.getInitType().getRank();
    resultOffsets.resize(initRank, zeroAttr);
    resultSizes = getDims(b, op->getLoc(), concreteOp.getInit());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    // Now, we still haven't implemented tiling init, so this function can't
    // work.
    return {};
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::ScatterOp concreteOp =
        cast<triton::linalg_ext::ScatterOp>(op);
    // Load indice and apply dim map to it.
    auto indexDepth = concreteOp.getIndexDepth();
    auto rank = concreteOp.getInitType().getRank();
    ArrayRef<int64_t> dimMap = concreteOp.getDimensionMap();

    auto batchNum = concreteOp.getBatchDimNum();
    SmallVector<Value> loadIndice{ivs.begin(), ivs.begin() + batchNum + 1};
    // Load update for loop 0 ~ rank+1.
    SmallVector<Value> loadUpdate{ivs.begin(), ivs.begin() + rank + batchNum};
    Value updateValue =
        b.create<memref::LoadOp>(loc, concreteOp.update(), loadUpdate);
    // Calculate actual index of input.
    SmallVector<Value> index{ivs.begin() + batchNum,
                             ivs.begin() + rank + batchNum};
    // StartPoint + Indice + LoadUpdate
    for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
      loadIndice.back() = b.create<arith::ConstantIndexOp>(loc, i);
      Value idx =
          b.create<memref::LoadOp>(loc, concreteOp.indice(), loadIndice);
      Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);
      auto dim = dimMap[i];
      // Add indice to start point.
      index[dim] = b.create<arith::AddIOp>(loc, ret, index[dim]);
    }
    SmallVector<Value> loadMask{ivs.begin(), ivs.begin() + batchNum};
    Value maskValue =
        b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 1));
    if (concreteOp.mask()) {
      maskValue = b.create<memref::LoadOp>(loc, concreteOp.mask(), loadMask);
    }
    // If we don't need to split init, since we already has index, just load and
    // calculate.
    b.create<scf::IfOp>(loc, maskValue, [&](OpBuilder &builder, Location loc) {
      Value initValue =
          builder.create<memref::LoadOp>(loc, concreteOp.getInit(), index);
      IRMapping bvm;
      Block &block = concreteOp.getRegion().front();
      bvm.map(block.getArgument(0), updateValue);
      bvm.map(block.getArgument(1), initValue);
      for (auto &blockOp : block.without_terminator()) {
        builder.clone(blockOp, bvm);
      }
      // The last op is yield op. Store the operand to
      // destination.
      builder.create<memref::StoreOp>(
          loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
          concreteOp.getInit(), index);
      builder.create<scf::YieldOp>(loc);
    });
    return success();
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::GatherOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::GatherOp>,
          triton::linalg_ext::GatherOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<mlir::DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::GatherOp concreteOp =
        cast<triton::linalg_ext::GatherOp>(op);
    SmallVector<utils::IteratorType> loops(concreteOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    Location loc = op->getLoc();
    triton::linalg_ext::GatherOp concreteOp =
        cast<triton::linalg_ext::GatherOp>(op);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim : llvm::seq<int64_t>(0, concreteOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, concreteOp.getInit(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::GatherOp concreteOp =
        cast<triton::linalg_ext::GatherOp>(op);
    SmallVector<int64_t> dimensionMap =
        llvm::to_vector(concreteOp.getDimensionMap());
    auto loc = op->getLoc();
    Value input = concreteOp.input();
    auto batchNum = concreteOp.getBatchDimNum();
    // Tile window.
    auto windowIndicesAndMask = tileByWindowSlice(
        b, loc, input, concreteOp.getInit(), concreteOp.indice(),
        concreteOp.mask(), dimensionMap, offsets, sizes, batchNum);
    // Slice of the input.
    // FIXME:Have to do this or tiling will core dump.
    auto inputRank = concreteOp.getInputType().getRank();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> inputStrides(inputRank, oneAttr);
    Value tiledInput;
    SmallVector<OpFoldResult> inputOffsets, inputSizes;
    inputOffsets = SmallVector<OpFoldResult>(inputRank, b.getIndexAttr(0));
    inputSizes = getDims(b, loc, input);
    tiledInput =
        getSimpliedSlice(b, loc, input, inputOffsets, inputSizes, inputStrides);
    assert(tiledInput && "failed to get slice of input");

    // Create tiled gather.
    triton::linalg_ext::GatherOp tiledGatherOp;
    if (windowIndicesAndMask[2]) {
      tiledGatherOp = b.create<triton::linalg_ext::GatherOp>(
          loc,
          ValueRange(
              {tiledInput, windowIndicesAndMask[1], windowIndicesAndMask[2]}),
          windowIndicesAndMask[0], dimensionMap, concreteOp.getRangedData(),
          concreteOp.getSignedIndice());
    } else {
      tiledGatherOp = b.create<triton::linalg_ext::GatherOp>(
          loc, ValueRange({tiledInput, windowIndicesAndMask[1]}),
          windowIndicesAndMask[0], dimensionMap, concreteOp.getRangedData(),
          concreteOp.getSignedIndice());
    }

    // Clean body region.
    auto &targetRegion = tiledGatherOp.getRegion();
    targetRegion.dropAllReferences();
    targetRegion.getBlocks().clear();

    // Clone body region from origin op.
    IRMapping map;
    concreteOp.getRegion().cloneInto(&targetRegion, map);
    return TilingResult{{tiledGatherOp},
                        SmallVector<Value>(tiledGatherOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = SmallVector<OpFoldResult>(offsets.begin(), offsets.end());
    resultSizes = SmallVector<OpFoldResult>(sizes.begin(), sizes.end());
    return success();
  }
  /// Tile window only.
  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::GatherOp concreteOp =
        cast<triton::linalg_ext::GatherOp>(op);
    // Load indice and apply dim map to it.
    auto indexDepth = concreteOp.getIndexDepth();
    auto rank = concreteOp.getInputType().getRank();
    ArrayRef<int64_t> dimMap = concreteOp.getDimensionMap();

    auto batchNum = concreteOp.getBatchDimNum();
    SmallVector<Value> loadIndice{ivs.begin(), ivs.begin() + batchNum + 1};
    SmallVector<Value> loadInit{ivs.begin(), ivs.begin() + rank + batchNum};
    // Load init for loop 0 ~ rank+1.
    Value initValue =
        b.create<memref::LoadOp>(loc, concreteOp.getInit(), loadInit);
    // Calculate actual index of input.
    SmallVector<Value> index{ivs.begin() + batchNum,
                             ivs.begin() + rank + batchNum};
    // StartPoint + Indice + LoadInit.
    for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
      loadIndice.back() = b.create<arith::ConstantIndexOp>(loc, i);
      Value idx =
          b.create<memref::LoadOp>(loc, concreteOp.indice(), loadIndice);
      Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);
      auto dim = dimMap[i];
      // Add indice to start point.
      index[dim] = b.create<arith::AddIOp>(loc, ret, index[dim]);
    }
    Value maskValue =
        b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 1));
    SmallVector<Value> loadMask{ivs.begin(), ivs.begin() + batchNum};
    if (concreteOp.mask()) {
      maskValue = b.create<memref::LoadOp>(loc, concreteOp.mask(), loadMask);
    }
    // If we don't need to split input, since we already has index, just load
    // and calculate.
    b.create<scf::IfOp>(loc, maskValue, [&](OpBuilder &builder, Location loc) {
      Value inputValue =
          builder.create<memref::LoadOp>(loc, concreteOp.input(), index);
      IRMapping bvm;
      Block &block = concreteOp.getRegion().front();
      bvm.map(block.getArgument(0), inputValue);
      bvm.map(block.getArgument(1), initValue);
      for (auto &blockOp : block.without_terminator()) {
        builder.clone(blockOp, bvm);
      }
      // The last op is yield op. Store the operand to
      // destination.
      builder.create<memref::StoreOp>(
          loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
          concreteOp.getInit(), loadInit);
      builder.create<scf::YieldOp>(loc);
    });
    return success();
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::AtomicCASOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::AtomicCASOp>,
          triton::linalg_ext::AtomicCASOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::AtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::AtomicCASOp>(op);
    SmallVector<utils::IteratorType> loops(atomicCASOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    triton::linalg_ext::AtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::AtomicCASOp>(op);
    Location loc = op->getLoc();
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, atomicCASOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, atomicCASOp.getInit(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::AtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::AtomicCASOp>(op);
    Location loc = atomicCASOp.getLoc();
    int64_t rank = atomicCASOp.getInitType().getRank();

    SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    auto inputSlice = b.create<tensor::ExtractSliceOp>(
        loc, atomicCASOp.input(), inputOffsets, inputSizes, strides);
    auto cmpSlice = b.create<tensor::ExtractSliceOp>(
        loc, atomicCASOp.cmp(), inputOffsets, inputSizes, strides);
    auto valSlice = b.create<tensor::ExtractSliceOp>(
        loc, atomicCASOp.val(), inputOffsets, inputSizes, strides);
    auto initSlice = b.create<tensor::ExtractSliceOp>(
        loc, atomicCASOp.getInit(), inputOffsets, inputSizes, strides);
    Operation *tiledOp = b.create<triton::linalg_ext::AtomicCASOp>(
        loc, initSlice.getType(), ValueRange({inputSlice, cmpSlice, valSlice}),
        initSlice, atomicCASOp.getMemoryOrder());
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
    if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return tilingResult;
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::AtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::AtomicCASOp>(op);
    OpBuilder::InsertionGuard guard(b);
    Value inputVal = b.create<memref::LoadOp>(loc, atomicCASOp.input(), ivs);
    Value cmpVal = b.create<memref::LoadOp>(loc, atomicCASOp.cmp(), ivs);
    Value valVal = b.create<memref::LoadOp>(loc, atomicCASOp.val(), ivs);

    b.create<memref::StoreOp>(loc, inputVal, atomicCASOp.getInit(), ivs);

    auto genericOp =
        b.create<memref::GenericAtomicRMWOp>(loc, atomicCASOp.getInit(), ivs);
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&genericOp.body().front(), b.getListener());

    Value currVal = genericOp.getCurrentValue();
    Value cmp;
    if (cmpVal.getType().isa<IntegerType>()) {
      cmp = bodyBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              currVal, cmpVal);
    } else if (cmpVal.getType().isa<FloatType>()) {
      cmp = bodyBuilder.create<mlir::arith::CmpFOp>(
          loc, arith::CmpFPredicate::OEQ, currVal, cmpVal);
    } else {
      return failure();
    }
    Value select =
        bodyBuilder.create<arith::SelectOp>(loc, cmp, valVal, currVal);
    bodyBuilder.create<memref::AtomicYieldOp>(loc, select);

    Value initVal = b.create<memref::LoadOp>(loc, atomicCASOp.getInit(), ivs);
    b.create<memref::StoreOp>(loc, initVal, atomicCASOp.input(), ivs);
    return success();
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::GatherAtomicCASOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::GatherAtomicCASOp>,
          triton::linalg_ext::GatherAtomicCASOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::GatherAtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::GatherAtomicCASOp>(op);
    SmallVector<utils::IteratorType> loops(atomicCASOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    triton::linalg_ext::GatherAtomicCASOp atomicCASOp =
        cast<triton::linalg_ext::GatherAtomicCASOp>(op);
    Location loc = op->getLoc();
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, atomicCASOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, atomicCASOp.getInit(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::GatherAtomicCASOp gatherAtomicCASOp =
        cast<triton::linalg_ext::GatherAtomicCASOp>(op);
    Location loc = gatherAtomicCASOp.getLoc();
    int64_t rank = gatherAtomicCASOp.getInitType().getRank();

    SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);

    auto cmpSlice = b.create<tensor::ExtractSliceOp>(
        loc, gatherAtomicCASOp.cmp(), inputOffsets, inputSizes, strides);
    auto valSlice = b.create<tensor::ExtractSliceOp>(
        loc, gatherAtomicCASOp.val(), inputOffsets, inputSizes, strides);
    auto indiceSlice = b.create<tensor::ExtractSliceOp>(
        loc, gatherAtomicCASOp.indice(), inputOffsets, inputSizes, strides);

    auto initSlice = b.create<tensor::ExtractSliceOp>(
        loc, gatherAtomicCASOp.getInit(), inputOffsets, inputSizes, strides);
    Operation *tiledOp = b.create<triton::linalg_ext::GatherAtomicCASOp>(
        loc, initSlice.getType(),
        ValueRange(
            {gatherAtomicCASOp.input(), cmpSlice, valSlice, indiceSlice}),
        initSlice, gatherAtomicCASOp.getMemoryOrder());
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);

    if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");
    return tilingResult;
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::GatherAtomicCASOp gatherAtomicCASOp =
        cast<triton::linalg_ext::GatherAtomicCASOp>(op);
    OpBuilder::InsertionGuard guard(b);

    Value indiceVal =
        b.create<memref::LoadOp>(loc, gatherAtomicCASOp.indice(), ivs);
    indiceVal = b.create<arith::IndexCastOp>(loc, b.getIndexType(), indiceVal);
    Value inputVal =
        b.create<memref::LoadOp>(loc, gatherAtomicCASOp.input(), indiceVal);
    Value cmpVal = b.create<memref::LoadOp>(loc, gatherAtomicCASOp.cmp(), ivs);
    Value valVal = b.create<memref::LoadOp>(loc, gatherAtomicCASOp.val(), ivs);

    b.create<memref::StoreOp>(loc, inputVal, gatherAtomicCASOp.getInit(), ivs);

    auto genericOp = b.create<memref::GenericAtomicRMWOp>(
        loc, gatherAtomicCASOp.getInit(), ivs);
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&genericOp.body().front(), b.getListener());

    Value currVal = genericOp.getCurrentValue();
    Value cmp;
    if (cmpVal.getType().isa<IntegerType>()) {
      cmp = bodyBuilder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              currVal, cmpVal);
    } else if (cmpVal.getType().isa<FloatType>()) {
      cmp = bodyBuilder.create<mlir::arith::CmpFOp>(
          loc, arith::CmpFPredicate::OEQ, currVal, cmpVal);
    } else {
      return failure();
    }
    Value select =
        bodyBuilder.create<arith::SelectOp>(loc, cmp, valVal, currVal);
    bodyBuilder.create<memref::AtomicYieldOp>(loc, select);

    Value initVal =
        b.create<memref::LoadOp>(loc, gatherAtomicCASOp.getInit(), ivs);
    b.create<memref::StoreOp>(loc, initVal, gatherAtomicCASOp.input(),
                              indiceVal);

    return success();
  }
};

// Contiguous atomicRMW
template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::AtomicRMWOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::AtomicRMWOp>,
          triton::linalg_ext::AtomicRMWOp> {
  /// Return the destination operands.
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::AtomicRMWOp concreteOp =
        cast<triton::linalg_ext::AtomicRMWOp>(op);
    SmallVector<utils::IteratorType> loops(concreteOp.getInputType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  /// Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op,
                                        OpBuilder &builder) const {
    Location loc = op->getLoc();
    triton::linalg_ext::AtomicRMWOp concreteOp =
        cast<triton::linalg_ext::AtomicRMWOp>(op);
    OpFoldResult zero = builder.getIndexAttr(0);
    OpFoldResult one = builder.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, concreteOp.getInputType().getRank())) {
      OpFoldResult ub = getDim(builder, loc, concreteOp.input(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::AtomicRMWOp atomicRMWOp =
        cast<triton::linalg_ext::AtomicRMWOp>(op);
    auto loc = op->getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);

    SmallVector<Value, 3> tiledInputs, tiledInits;
    // Use original src.
    tiledInits.push_back(atomicRMWOp.src());
    // Slice input and dst.
    Value input = atomicRMWOp.input();
    auto inputRank = input.getType().cast<ShapedType>().getRank();
    SmallVector<OpFoldResult> inputStrides(inputRank, oneAttr);
    Value tiledInput =
        getSimpliedSlice(b, loc, input, offsets, sizes, inputStrides);
    assert(tiledInput && "failed to get slice of input");
    tiledInputs.push_back(tiledInput);
    Value dst = atomicRMWOp.dst();
    Value tiledDst =
        getSimpliedSlice(b, loc, dst, offsets, sizes, inputStrides);
    assert(tiledDst && "failed to get slice of dst");
    tiledInits.push_back(tiledDst);

    // Create tiled atomic_rmw.
    triton::linalg_ext::AtomicRMWOp tiledAtomicRMWOp =
        b.create<triton::linalg_ext::AtomicRMWOp>(loc, tiledInputs, tiledInits,
                                                  atomicRMWOp.getAtomicType(),
                                                  atomicRMWOp.getMemoryOrder());

    return TilingResult{{tiledAtomicRMWOp},
                        SmallVector<Value>(tiledAtomicRMWOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    triton::linalg_ext::AtomicRMWOp atomicRMWOp =
        cast<triton::linalg_ext::AtomicRMWOp>(op);
    // Result 0 is src, we keep it unchanged.

    if (resultNumber == 0) {
      auto zeroAttr = b.getI64IntegerAttr(0);
      auto initRank = atomicRMWOp.getSrcType().getRank();
      auto initShape = atomicRMWOp.getSrcType().getShape();
      for (unsigned r = 0; r < initRank; ++r) {
        if (!isNoTile(sizes[r], offsets[r], initShape, r)) {
          return failure();
        }
      }
      resultOffsets.clear();
      resultOffsets.append(offsets.begin(), offsets.end());
      resultSizes.clear();
      resultSizes.append(sizes.begin(), sizes.end());
      return success();
    }
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    assert((resultNumber == 1) && "only support tile dst now.");
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
    if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return tilingResult;
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::AtomicRMWOp concreteOp =
        cast<triton::linalg_ext::AtomicRMWOp>(op);

    OpBuilder::InsertionGuard guard(b);
    auto inputRank = concreteOp.getInputType().getRank();
    auto srcRank = concreteOp.getSrcType().getRank();
    // Get Indices.
    SmallVector<Value> index{ivs.begin() + 1, ivs.begin() + srcRank + 1};
    for (auto i : llvm::seq<unsigned>(0, inputRank)) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, i);
      Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);
      // Add indice to start point.
      index[i] = b.create<arith::AddIOp>(loc, ret, index[i]);
    }
    // Get Input.
    auto valueRank = concreteOp.getInputType().getRank();
    SmallVector<Value> loadInput{ivs.begin(), ivs.begin() + valueRank};
    Value inputValue =
        b.create<memref::LoadOp>(loc, concreteOp.input(), loadInput);

    // Get rmw Value.
    Value rmwValue = b.create<memref::LoadOp>(loc, concreteOp.src(), index);
    b.create<memref::StoreOp>(loc, rmwValue, concreteOp.dst(), loadInput);
    if (failed(createMemrefAtomicRMW(b, loc, inputValue, concreteOp.src(),
                                     index, concreteOp.getAtomicType())))
      return failure();

    return success();
  }
};

// Discrete atomicRMW.
template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::GatherAtomicRMWOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::GatherAtomicRMWOp>,
          triton::linalg_ext::GatherAtomicRMWOp> {
  /// Return the destination operands.
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::GatherAtomicRMWOp concreteOp =
        cast<triton::linalg_ext::GatherAtomicRMWOp>(op);
    SmallVector<utils::IteratorType> loops(concreteOp.getInputType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  /// Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op,
                                        OpBuilder &builder) const {
    Location loc = op->getLoc();
    triton::linalg_ext::GatherAtomicRMWOp concreteOp =
        cast<triton::linalg_ext::GatherAtomicRMWOp>(op);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, concreteOp.getInputType().getRank())) {
      OpFoldResult ub = getDim(builder, loc, concreteOp.input(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::GatherAtomicRMWOp atomicRMWOp =
        cast<triton::linalg_ext::GatherAtomicRMWOp>(op);
    auto loc = op->getLoc();
    auto zeroAttr = b.getI64IntegerAttr(0);
    auto oneAttr = b.getI64IntegerAttr(1);

    SmallVector<Value, 3> tiledInputs, tiledInits;
    // Use original src.
    tiledInits.push_back(atomicRMWOp.src());
    // Slice input and window batch.
    Value input = atomicRMWOp.input();
    auto inputRank = input.getType().cast<ShapedType>().getRank();
    SmallVector<OpFoldResult> inputStrides(inputRank, oneAttr);
    Value tiledInput =
        getSimpliedSlice(b, loc, input, offsets, sizes, inputStrides);
    assert(tiledInput && "failed to get slice of input");
    tiledInputs.push_back(tiledInput);
    Value window = atomicRMWOp.window();
    Value tiledWindow =
        getSimpliedSlice(b, loc, window, offsets, sizes, inputStrides);
    assert(tiledWindow && "failed to get slice of window");
    tiledInits.push_back(tiledWindow);
    // Slice indice.
    auto indice = atomicRMWOp.indice();
    auto indiceRank = indice.getType().cast<ShapedType>().getRank();
    SmallVector<OpFoldResult> indiceOffsets(indiceRank, zeroAttr);
    SmallVector<OpFoldResult> indiceSizes(indiceRank);
    SmallVector<OpFoldResult> indiceStrides(indiceRank, oneAttr);
    indiceOffsets[0] = offsets[0];
    indiceSizes[0] = sizes[0];
    for (auto dim : llvm::seq<int64_t>(1, indiceRank)) {
      indiceSizes[dim] = getDim(b, loc, indice, dim);
    }
    // Slice indice batch.
    Value tiledIndice = getSimpliedSlice(b, loc, indice, indiceOffsets,
                                         indiceSizes, indiceStrides);
    assert(tiledIndice && "failed to get batch slice of indice");
    // Create add offsets.
    SmallVector<Value> addOffsetVals;
    auto indiceEleTy = getElementTypeOrSelf(indice.getType());
    for (auto dim : llvm::seq<int64_t>(1, inputRank)) {
      Value val = b.create<tensor::EmptyOp>(loc, int64_t{1}, indiceEleTy);
      Value offset = getValueOrCreateConstantIndexOp(b, loc, offsets[dim]);
      Value castOffset = b.create<arith::IndexCastOp>(loc, indiceEleTy, offset);
      Value fill = b.create<linalg::FillOp>(loc, castOffset, val).result();
      addOffsetVals.push_back(fill);
    }

    // Insert add offsets to empty buffer.
    Value result =
        b.create<tensor::EmptyOp>(loc, int64_t{inputRank - 1}, indiceEleTy);
    auto zero = b.getIndexAttr(0);
    auto one = b.getIndexAttr(1);
    SmallVector<OpFoldResult> insertOffsets(1, zero);
    SmallVector<OpFoldResult> insertStrides(1, one);
    SmallVector<OpFoldResult> insertSizes;
    for (auto addOffsetVal : addOffsetVals) {
      insertSizes = getDims(b, loc, addOffsetVal);
      result = b.createOrFold<tensor::InsertSliceOp>(
          loc, addOffsetVal, result, insertOffsets, insertSizes, insertStrides);
      insertOffsets[0] = b.createOrFold<arith::AddIOp>(
          loc, materializeOpFoldResult(b, loc, insertOffsets[0]),
          materializeOpFoldResult(b, loc, insertSizes[0]));
    }
    // Broadcast add offsets.
    SmallVector<OpFoldResult> addOffsetBroadShape{
        sizes[0], b.getIndexAttr(inputRank - 1)};
    Value broadVal = addBroadcast(b, loc, result, addOffsetBroadShape, {0});

    // Add offset to non-batch indice.
    tiledIndice = b.createOrFold<arith::AddIOp>(loc, tiledIndice, broadVal);
    tiledInputs.push_back(tiledIndice);
    // Slice mask.
    Value mask = atomicRMWOp.mask();
    if (mask) {
      SmallVector<OpFoldResult> maskOffsets{offsets[0]};
      SmallVector<OpFoldResult> maskSizes{sizes[0]};
      SmallVector<OpFoldResult> maskStrides{oneAttr};
      Value tiledMask =
          getSimpliedSlice(b, loc, mask, maskOffsets, maskSizes, maskStrides);
      tiledInputs.push_back(tiledMask);
      assert(tiledMask && "failed to get slice of mask");
    }
    // Create tiled atomic_rmw.
    triton::linalg_ext::GatherAtomicRMWOp tiledAtomicRMWOp =
        b.create<triton::linalg_ext::GatherAtomicRMWOp>(
            loc, tiledInputs, tiledInits, atomicRMWOp.getAtomicType(),
            atomicRMWOp.getMemoryOrder());

    return TilingResult{{tiledAtomicRMWOp},
                        SmallVector<Value>(tiledAtomicRMWOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    triton::linalg_ext::GatherAtomicRMWOp atomicRMWOp =
        cast<triton::linalg_ext::GatherAtomicRMWOp>(op);
    // Result 0 is src, we keep it unchanged.
    if (resultNumber == 0) {
      auto zeroAttr = b.getI64IntegerAttr(0);
      auto initRank = atomicRMWOp.getSrcType().getRank();
      resultOffsets.resize(initRank, zeroAttr);
      resultSizes = getDims(b, op->getLoc(), atomicRMWOp.src());
      return success();
    }
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    if (resultNumber != 1) {
      return op->emitOpError("only support tile window now.");
    }
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
    if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");

    return tilingResult;
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::GatherAtomicRMWOp concreteOp =
        cast<triton::linalg_ext::GatherAtomicRMWOp>(op);

    OpBuilder::InsertionGuard guard(b);
    auto indexDepth = concreteOp.getIndexDepth();
    auto srcRank = concreteOp.getSrcType().getRank();
    // Get Indices.
    SmallVector<Value> loadIndice{ivs.front(), Value()};
    SmallVector<Value> index{ivs.begin() + 1, ivs.begin() + srcRank + 1};
    for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
      loadIndice.back() = b.create<arith::ConstantIndexOp>(loc, i);
      Value idx =
          b.create<memref::LoadOp>(loc, concreteOp.indice(), loadIndice);
      Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);
      // Add indice to start point.
      index[i] = b.create<arith::AddIOp>(loc, ret, index[i]);
    }

    // Get Mask.
    if (concreteOp.mask()) {
      Value one =
          b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI8Type(), 1));
      Value maskValue =
          b.create<memref::LoadOp>(loc, concreteOp.mask(), ivs.front());
      maskValue = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          maskValue, one);

      auto ifOp = b.create<scf::IfOp>(loc, maskValue, /*else=*/false);
      b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    // Get Input.
    auto valueRank = concreteOp.getInputType().getRank();
    SmallVector<Value> loadInput{ivs.begin(), ivs.begin() + valueRank};
    Value inputValue =
        b.create<memref::LoadOp>(loc, concreteOp.input(), loadInput);

    // Get rmw Value.
    Value rmwValue = b.create<memref::LoadOp>(loc, concreteOp.src(), index);
    b.create<memref::StoreOp>(loc, rmwValue, concreteOp.window(), loadInput);
    if (failed(createMemrefAtomicRMW(b, loc, inputValue, concreteOp.src(),
                                     index, concreteOp.getAtomicType())))
      return failure();

    return success();
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::PadOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::PadOp>,
          triton::linalg_ext::PadOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::PadOp concreteOp = cast<triton::linalg_ext::PadOp>(op);
    SmallVector<utils::IteratorType> loops(concreteOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    Location loc = op->getLoc();
    triton::linalg_ext::PadOp concreteOp = cast<triton::linalg_ext::PadOp>(op);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> ranges;
    for (auto dim : llvm::seq<int64_t>(0, concreteOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, concreteOp.getInit(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  /// Instantiate the tiled implementation of the padOp.
  /// For padOp, tile `init` according to the parameters `offsets` and `sizes`,
  /// and then determine the `input`, `low`, and `high` of each pad slice based
  /// on the positional relationship between `input` and `init`. The calculation
  /// rules of `input`, `low` and `high` for each dimension are the same and
  /// independent. Taking the i-th dimension as an example, the symbols involved
  /// and their representations are as shown in the table below:
  ///
  /// +-------------+----------------------------------------------------------+
  /// |   symbol    |             representation                               |
  /// +-------------+----------------------------------------------------------+
  /// |     S       | The input size of this dimension.                        |
  /// +-------------+----------------------------------------------------------+
  /// |     L       | The padding size along the start of this dimension.      |
  /// +-------------+----------------------------------------------------------+
  /// |     H       | The padding size along the end of this dimension.        |
  /// +-------------+----------------------------------------------------------+
  /// |  srcStart   | The start point of the input.                            |
  /// +-------------+----------------------------------------------------------+
  /// |  srcEnd     | The end point of the input.                              |
  /// +-------------+----------------------------------------------------------+
  /// | curDstStart | The start point of the current slice of init.            |
  /// +-------------+----------------------------------------------------------+
  /// | curDstEnd   | The end point of the current slice of init.              |
  /// +-------------+----------------------------------------------------------+
  /// | curSrcOffset| The start point of the current slice of input.           |
  /// +-------------+----------------------------------------------------------+
  /// | curSrcSize  | The size of the current slice of input.                  |
  /// +-------------+----------------------------------------------------------+
  /// | curLow      | The current slice's padding size along the start of dim. |
  /// +-------------+----------------------------------------------------------+
  /// | curHigh     | The current slice's padding size along the end of dim.   |
  /// +-------------+----------------------------------------------------------+
  ///
  /// So the size of `init` on this dimension is L+S+H. Use the `init` of the
  /// current dimension as the coordinate axis, it can be derived:
  ///   srcStart = L;  srcEnd = L + S;
  ///   curDstStart = offset; curDstEnd = offset + size;
  ///
  /// It can be classified into the following cases according to the positions
  /// of `curDstStart` and `curDstEnd` to calculate the `curSrcOffset`,
  /// `curSrcSize`, `curLow` and `curHigh`:
  /// +---+----------------------------------+---------------------------------+
  /// |   |       curDstStart's pos          |    curDstEnd's pos              |
  /// +---+----------------------------------+---------------------------------+
  /// | 1 |                                  |     curDstEnd < srcStart        |
  /// +---+                                  +---------------------------------+
  /// | 2 |       curDstStart < srcStart     |     curDstEnd > srcEnd          |
  /// +---+                                  +---------------------------------+
  /// | 3 |                                  | srcStart <= curDstEnd <= srcEnd |
  /// +---+----------------------------------+---------------------------------+
  /// | 4 |       curDstStart > srcEnd       |       curDstEnd > srcEnd        |
  /// +---+----------------------------------+---------------------------------+
  /// | 5 |                                  |       curDstEnd <= srcEnd |
  /// +---+  srcStart<=curDstStart<=srcEnd   +---------------------------------+
  /// | 6 |                                  |       curDstEnd > srcEnd       |
  /// +---+----------------------------------+---------------------------------+
  ///
  /// The calculation results under different cases are as follows:
  /// +---+-------------+--------------------+-------------+-------------------+
  /// |   | curSrcOffset|   curSrcSize       |   curLow    |    curHigh        |
  /// +---+-------------+--------------------+-------------+-------------------+
  /// | 1 |             |   0                |     0       |    size           |
  /// +---+             +--------------------+-------------+-------------------+
  /// | 2 |   0         |   S                | srcStart    | curDstEnd-srcEnd  |
  /// +---+             +--------------------+    -        +-------------------+
  /// | 3 |             |curDstStart-srcStart| curDstStart |    0              |
  /// +---+-------------+--------------------+-------------+-------------------+
  /// | 4 |   S         |   0                |     0       |    size           |
  /// +---+-------------+--------------------+-------------+-------------------+
  /// | 5 | curDstStart |   size             |             |    0              |
  /// +---+    -        +--------------------+     0       +-------------------+
  /// | 6 | srcStart    | srcEnd-curDstStart |             | curDstEnd-srcEnd  |
  /// +---+-------------+--------------------+-------------+-------------------+
  ///
  /// After simplification, it can be derived:
  ///   curSrcOffset = case1-3 ? 0 : curDstStart - srcStart;
  ///   curSrcOffset = case4   ? S : curSrcOffset;
  ///
  ///   curSrcSize = case1 ? 0 : curDstStart -  srcStart
  ///   curSrcSize = case2 ? S : curSrcSize
  ///   curSrcSize = case4 ? 0 : curSrcSize
  ///   curSrcSize = case5 ? size : curSrcSize
  ///   curSrcSize = case6 ? srcEnd - curDstStart : curSrcSize
  ///   curSrcSize = min (curSrcSize, size)
  ///
  ///   curLow = case2-3? srcStart - curDstStart : 0
  ///
  ///   curHigh = case1 ? size : 0
  ///   curHigh = case2 ? curDstEnd - srcEnd : curHigh
  ///   curHigh = case4 ? size : curHigh
  ///   curHigh = case6 ? curDstEnd - srcEnd : curHigh
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    Location loc = op->getLoc();
    triton::linalg_ext::PadOp concreteOp = cast<triton::linalg_ext::PadOp>(op);
    auto oneAttr = b.getI64IntegerAttr(1);
    auto originLows = concreteOp.getMixedLowPad();
    auto originHighs = concreteOp.getMixedHighPad();
    auto initType = concreteOp.getInitType();
    int64_t rank = initType.getRank();
    // Slice of the init.
    Value init = concreteOp.getInit();
    SmallVector<OpFoldResult> initStrides(rank, oneAttr);
    SmallVector<OpFoldResult> initOffsets =
        SmallVector<OpFoldResult>(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> initSizes =
        SmallVector<OpFoldResult>(sizes.begin(), sizes.end());
    Value tiledInit =
        getSlice(b, loc, init, initOffsets, initSizes, initStrides);
    assert(tiledInit && "failed to get slice of init");

    // Slice of the Input.
    auto input = concreteOp.getDpsInputOperand(0)->get();
    SmallVector<OpFoldResult> inputStrides(rank, oneAttr);
    SmallVector<OpFoldResult> inputOffsets, inputSizes, lows, highs;
    lows.reserve(rank);
    highs.reserve(rank);
    Value zeroIndex = b.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned r = 0; r < rank; ++r) {
      auto tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[r]);
      auto tileOffset = getValueOrCreateConstantIndexOp(b, loc, offsets[r]);
      Value originLowIndex =
          getValueOrCreateConstantIndexOp(b, loc, originLows[r]);
      Value originHighIndex =
          getValueOrCreateConstantIndexOp(b, loc, originHighs[r]);
      if (isNoTile(tileSize, tileOffset, initType.getShape(), r)) {
        inputOffsets.push_back(b.getIndexAttr(0));
        Value dim = getDimValue(b, loc, input, r);
        inputSizes.push_back(getAsOpFoldResult(dim));
        lows.push_back(getAsOpFoldResult(originLowIndex));
        highs.push_back(getAsOpFoldResult(originHighIndex));
        continue;
      }
      Value inputDimSize = getDimValue(b, loc, input, r);
      Value srcStart = originLowIndex;
      Value srcEnd = b.create<arith::AddIOp>(loc, srcStart, inputDimSize);
      Value curDstStart = tileOffset;
      Value curDstEnd = b.create<arith::AddIOp>(loc, curDstStart, tileSize);

      // Classify the position of curDstStart.
      // This is case 1, 2, 3: curDstStart < srcStart.
      Value curDstStartBeforeSrc = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, curDstStart, srcStart);
      // This is case 4: curDstStart > srcStart.
      Value curDstStartAfterSrc = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, curDstStart, srcEnd);
      // This is case 5, 6: srcStart <= curDstStart <= srcEnd.
      Value curDstStartInSrc = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, curDstStart,
                                  srcStart),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, curDstStart,
                                  srcEnd));

      // Classify the position of curDstEnd.
      // This is case 1: curDstEnd < srcStart.
      Value curDstEndBeforeSrc = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, curDstEnd, srcStart);
      // This is case 2, 4 or 6: curDstEnd > srcEnd.
      Value curDstEndAfterSrc = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, curDstEnd, srcEnd);
      // curDstEnd >= srcStart.
      Value curDstEndGESrcStart = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, curDstEnd, srcStart);
      // curDstEnd <= srcEnd.
      Value curDstEndLESrcEnd = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, curDstEnd, srcEnd);

      Value curSrcOffset, curSrcSize, curLow, curHigh;

      // curSrcOffset = case1-3 ? 0 : curDstStart - srcStart.
      Value offset1 = b.create<arith::SubIOp>(loc, curDstStart, srcStart);
      curSrcOffset = b.create<arith::SelectOp>(loc, curDstStartBeforeSrc,
                                               zeroIndex, offset1);
      // curSrcOffset = case4 ? S : curSrcOffset.
      curSrcOffset = b.create<arith::SelectOp>(loc, curDstStartAfterSrc,
                                               inputDimSize, curSrcOffset);
      inputOffsets.push_back(getAsOpFoldResult(curSrcOffset));

      Value size1 = b.create<arith::SubIOp>(loc, curDstEnd, srcStart);
      Value size2 = b.create<arith::SubIOp>(loc, srcEnd, curDstStart);
      // This is case 1: (curDstStart < srcStart) && (curDstEnd < srcStart).
      Value curDstBeforeSrc = b.create<arith::AndIOp>(loc, curDstStartBeforeSrc,
                                                      curDstEndBeforeSrc);
      // This is case 2: (curDstStart < srcStart) && (curDstEnd > srcEnd).
      Value srcInCurDst =
          b.create<arith::AndIOp>(loc, curDstStartBeforeSrc, curDstEndAfterSrc);
      // This is case 5: (srcStart <= curDstStart <= srcEnd) && (curDstEnd <=
      // srcEnd).
      Value curDstInSrc =
          b.create<arith::AndIOp>(loc, curDstStartInSrc, curDstEndLESrcEnd);
      // This is case 6: (srcStart <= curDstStart <= srcEnd) && (curDstEnd >
      // srcEnd).
      Value curDstStartInSrcAndcurDstEndAfterSrc =
          b.create<arith::AndIOp>(loc, curDstStartInSrc, curDstEndAfterSrc);
      // curSrcSize = case1 ? 0 : curDstStart -  srcStart.
      curSrcSize =
          b.create<arith::SelectOp>(loc, curDstBeforeSrc, zeroIndex, size1);
      // curSrcSize = case2 ? S : curSrcSize.
      curSrcSize =
          b.create<arith::SelectOp>(loc, srcInCurDst, inputDimSize, curSrcSize);
      // curSrcSize = case4 ? 0 : curSrcSize.
      curSrcSize = b.create<arith::SelectOp>(loc, curDstStartAfterSrc,
                                             zeroIndex, curSrcSize);
      // curSrcSize = case5 ? size : curSrcSize.
      curSrcSize =
          b.create<arith::SelectOp>(loc, curDstInSrc, tileSize, curSrcSize);
      // curSrcSize = case6 ? srcEnd - curDstStart : curSrcSize.
      curSrcSize = b.create<arith::SelectOp>(
          loc, curDstStartInSrcAndcurDstEndAfterSrc, size2, curSrcSize);
      // curSrcSize = min (curSrcSize, size).
      curSrcSize = b.create<arith::MinSIOp>(loc, curSrcSize, tileSize);
      inputSizes.push_back(getAsOpFoldResult(curSrcSize));

      Value low1 = b.create<arith::SubIOp>(loc, srcStart, curDstStart);
      // This is case 2, 3: (curDstStart < srcStart) && (curDstEnd >= srcStart).
      Value cond = b.create<arith::AndIOp>(loc, curDstStartBeforeSrc,
                                           curDstEndGESrcStart);
      // curLow = case2-3? srcStart - curDstStart : 0.
      curLow = b.create<arith::SelectOp>(loc, cond, low1, zeroIndex);
      lows.push_back(getAsOpFoldResult(curLow));

      Value high1 = b.create<arith::SubIOp>(loc, curDstEnd, srcEnd);
      // curHigh = case1 ? size : 0.
      curHigh =
          b.create<arith::SelectOp>(loc, curDstBeforeSrc, tileSize, zeroIndex);
      // curHigh = case2 ? curDstEnd - srcEnd : curHigh.
      curHigh = b.create<arith::SelectOp>(loc, srcInCurDst, high1, curHigh);
      // curHigh = case4 ? size : curHigh.
      curHigh = b.create<arith::SelectOp>(loc, curDstStartAfterSrc, tileSize,
                                          curHigh);
      // curHigh = case6 ? curDstEnd - srcEnd : curHigh.
      curHigh = b.create<arith::SelectOp>(
          loc, curDstStartInSrcAndcurDstEndAfterSrc, high1, curHigh);
      highs.push_back(getAsOpFoldResult(curHigh));
    }
    Value tiledInput =
        getSlice(b, loc, input, inputOffsets, inputSizes, inputStrides);
    assert(tiledInput && "failed to get slice of input");

    auto tiledPad = b.create<triton::linalg_ext::PadOp>(
        loc, tiledInput, tiledInit, concreteOp.getPvalue(), lows, highs);
    return TilingResult{{tiledPad}, SmallVector<Value>(tiledPad->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = SmallVector<OpFoldResult>(offsets.begin(), offsets.end());
    resultSizes = SmallVector<OpFoldResult>(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    return tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::PadOp concreteOp = cast<triton::linalg_ext::PadOp>(op);
    auto input = concreteOp.input();
    auto init = concreteOp.getInit();
    auto pvalue = concreteOp.getPvalue();
    auto staticLow = concreteOp.getStaticLow();
    auto staticHigh = concreteOp.getStaticHigh();
    int64_t rank = concreteOp.getInitType().getRank();
    SmallVector<Value> loadInit{ivs.begin(), ivs.begin() + rank};
    SmallVector<Value, 4> inputIndexs;
    Value isOutRange = b.create<arith::ConstantOp>(loc, b.getBoolAttr(false));
    for (int i = 0; i < rank; i++) {
      Value ub = getDimValue(b, loc, init, i);
      Value lowIndex = b.create<arith::ConstantIndexOp>(loc, staticLow[i]);
      Value highIndex = b.create<arith::ConstantIndexOp>(loc, staticHigh[i]);
      Value upBound = b.create<arith::SubIOp>(loc, ub, highIndex);
      Value isLTLow = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              ivs[i], lowIndex);
      Value isGEHigh = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               ivs[i], upBound);
      Value pred = b.create<arith::OrIOp>(loc, isLTLow, isGEHigh);
      isOutRange = b.create<arith::OrIOp>(loc, pred, isOutRange);
      // Calculate input coordinate.
      Value index = b.create<arith::SubIOp>(loc, ivs[i], lowIndex);
      inputIndexs.push_back(index);
    }
    b.create<scf::IfOp>(
        loc, isOutRange,
        [&](OpBuilder &b, Location loc) {
          b.create<memref::StoreOp>(loc, pvalue, init, loadInit);
          b.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &b, Location loc) {
          Value inputValue = b.create<memref::LoadOp>(loc, input, inputIndexs);
          b.create<memref::StoreOp>(loc, inputValue, init, loadInit);
          b.create<scf::YieldOp>(loc);
        });
    return success();
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::AssertOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::AssertOp>,
          triton::linalg_ext::AssertOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::AssertOp assertOp =
        cast<triton::linalg_ext::AssertOp>(op);
    SmallVector<utils::IteratorType> loops(assertOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    triton::linalg_ext::AssertOp assertOp =
        cast<triton::linalg_ext::AssertOp>(op);
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Range> ranges;
    for (auto dim : llvm::seq<int64_t>(0, assertOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, assertOp.getCondition(), dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::AssertOp assertOp =
        cast<triton::linalg_ext::AssertOp>(op);
    Location loc = assertOp.getLoc();
    int64_t rank = assertOp.getInitType().getRank();

    SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    auto inputSlice = b.create<tensor::ExtractSliceOp>(
        loc, assertOp.getCondition(), inputOffsets, inputSizes, strides);

    Operation *tiledOp = b.create<triton::linalg_ext::AssertOp>(
        loc, inputSlice.getType(), inputSlice, assertOp.getMsg());
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    return tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    llvm_unreachable("No scalar implementation for assert op");
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::LibdeviceCallOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::LibdeviceCallOp>,
          triton::linalg_ext::LibdeviceCallOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::LibdeviceCallOp libdeviceCallOp =
        cast<triton::linalg_ext::LibdeviceCallOp>(op);
    SmallVector<utils::IteratorType> loops(
        libdeviceCallOp.getInitType().getRank(), utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    triton::linalg_ext::LibdeviceCallOp libdeviceCallOp =
        cast<triton::linalg_ext::LibdeviceCallOp>(op);
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Range> ranges;
    for (auto dim :
         llvm::seq<int64_t>(0, libdeviceCallOp.getInitType().getRank())) {
      OpFoldResult ub = getDim(b, loc, libdeviceCallOp.inputs()[0], dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::LibdeviceCallOp libdeviceCallOp =
        cast<triton::linalg_ext::LibdeviceCallOp>(op);
    Location loc = libdeviceCallOp.getLoc();
    int64_t rank = libdeviceCallOp.getInitType().getRank();

    SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    SmallVector<Value> inputsSlice;
    for (auto input : libdeviceCallOp.inputs()) {
      if (input.getType().isa<ShapedType>()) {
        auto inputSlice = b.create<tensor::ExtractSliceOp>(
            loc, input, inputOffsets, inputSizes, strides);
        inputsSlice.push_back(inputSlice);
      } else {
        inputsSlice.push_back(input);
      }
    }
    auto initSlice = b.create<tensor::ExtractSliceOp>(
        loc, libdeviceCallOp.getInit(), inputOffsets, inputSizes, strides);
    Operation *tiledOp = b.create<triton::linalg_ext::LibdeviceCallOp>(
        loc, ValueRange(inputsSlice), initSlice, libdeviceCallOp.getSymbol());
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    return tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    llvm_unreachable(
        "No scalar implementation for linalg_ext.libdevice_call op");
  }
};

template <>
struct LinalgExtOpTilingInterface<triton::linalg_ext::ScanOp>
    : public TilingInterface::ExternalModel<
          LinalgExtOpTilingInterface<triton::linalg_ext::ScanOp>,
          triton::linalg_ext::ScanOp> {

  SmallVector<Value> getDestinationOperands(Operation *op,
                                            OpBuilder &builder) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::linalg_ext::ScanOp concreteOp =
        cast<triton::linalg_ext::ScanOp>(op);
    int64_t operandRank = concreteOp.getOperandRank();
    SmallVector<utils::IteratorType> iteratorTypes(
        operandRank, utils::IteratorType::parallel);
    iteratorTypes[concreteOp.getDimensions()[0]] =
        utils::IteratorType::reduction;
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op,
                                        OpBuilder &builder) const {
    triton::linalg_ext::ScanOp concreteOp =
        cast<triton::linalg_ext::ScanOp>(op);
    int64_t operandRank = concreteOp.getOperandRank();
    SmallVector<Range> loopBounds;
    Location loc = concreteOp.getLoc();
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value source = concreteOp.inputs()[0];
    for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
      OpFoldResult ub = getDim(builder, loc, source, dim);
      loopBounds.emplace_back(Range{zero, ub, one});
    }
    return loopBounds;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &builder,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::ScanOp concreteOp =
        cast<triton::linalg_ext::ScanOp>(op);
    // FIXME: support multi operands in tiling interface.
    if ((concreteOp.getNumDpsInputs() != 1) ||
        (concreteOp.getNumDpsInits() != 2))
      return op->emitOpError("tiling interface only support single input now.");
    int64_t rank = concreteOp.getOperandRank();
    assert(offsets.size() == static_cast<size_t>(rank) &&
           sizes.size() == static_cast<size_t>(rank));
    auto oneAttr = builder.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    SmallVector<Value> tiledOperands;
    tiledOperands.emplace_back(getSimpliedSlice(builder, concreteOp.getLoc(),
                                                concreteOp.inputs()[0], offsets,
                                                sizes, strides));
    tiledOperands.emplace_back(getSimpliedSlice(builder, concreteOp.getLoc(),
                                                concreteOp.outputs()[0],
                                                offsets, sizes, strides));
    if (rank > 1) {
      SmallVector<OpFoldResult> initOffsets, initSizes;
      if (failed(getResultTilePosition(op, builder, 1, offsets, sizes,
                                       initOffsets, initSizes))) {
        return {};
      }
      SmallVector<OpFoldResult> initStrides(rank - 1, oneAttr);
      tiledOperands.emplace_back(
          getSimpliedSlice(builder, concreteOp.getLoc(), concreteOp.inits()[0],
                           initOffsets, initSizes, initStrides));
    } else {
      tiledOperands.emplace_back(concreteOp.inits()[0]);
    }

    SmallVector<Type, 4> resultTypes;
    if (concreteOp.hasPureTensorSemantics()) {
      resultTypes.push_back(tiledOperands[1].getType());
      resultTypes.push_back(tiledOperands[2].getType());
    }

    Operation *tiledScanOp = mlir::clone(builder, concreteOp.getOperation(),
                                         resultTypes, tiledOperands);
    return TilingResult{{tiledScanOp},
                        SmallVector<Value>(tiledScanOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &builder,
                        unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    triton::linalg_ext::ScanOp concreteOp =
        cast<triton::linalg_ext::ScanOp>(op);
    if (resultNumber == 0) {
      resultOffsets.assign(offsets.begin(), offsets.end());
      resultSizes.assign(sizes.begin(), sizes.end());
      return success();
    }
    if (resultNumber == 1) {
      int64_t rank = concreteOp.getOperandRank();
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == concreteOp.getDimensions()[0]) {
          continue;
        }
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
      return success();
    }
    return failure();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &builder,
                          unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    triton::linalg_ext::ScanOp scanOp = cast<triton::linalg_ext::ScanOp>(op);
    SmallVector<OpFoldResult> domainOffsets, domainSizes;
    domainOffsets.assign(offsets.begin(), offsets.end());
    domainSizes.assign(sizes.begin(), sizes.end());
    if (resultNumber == 1) {
      int64_t dim = scanOp.getDimensions()[0];
      auto inputSizes = getDims(builder, op->getLoc(), scanOp.inputs()[0]);
      domainOffsets.insert(domainOffsets.begin() + dim,
                           builder.getIndexAttr(0));
      domainSizes.insert(domainSizes.begin() + dim, inputSizes[dim]);
    }
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    FailureOr<TilingResult> tilingResult =
        tilingInterfaceOp.getTiledImplementation(builder, domainOffsets,
                                                 domainSizes);
    if (failed(tilingResult) || tilingResult->tiledOps.size() != 1)
      return op->emitOpError("failed to generate tiled implementation");
    return tilingResult;
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    triton::linalg_ext::ScanOp concreteOp =
        cast<triton::linalg_ext::ScanOp>(op);
    // FIXME: support multi operands in tiling interface.
    if ((concreteOp.getNumDpsInputs() != 1) ||
        (concreteOp.getNumDpsInits() != 2))
      return op->emitOpError("tiling interface only support single input now.");
    SmallVector<Value> indices, scanBlkArgs;
    indices.append(ivs.begin(), ivs.end());
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    int64_t scanDim = concreteOp.getDimensions()[0];
    Value size = getDimValue(b, loc, concreteOp.inputs()[0], scanDim);
    size = b.create<arith::SubIOp>(loc, size, one);
    if (concreteOp.getReverse())
      indices[scanDim] = b.create<arith::SubIOp>(loc, size, indices[scanDim]);
    SmallVector<Value> accIndices;
    for (int i = 0; i < indices.size(); i++) {
      if (i != scanDim)
        accIndices.push_back(indices[i]);
    }
    scanBlkArgs.push_back(
        b.create<memref::LoadOp>(loc, concreteOp.inputs()[0], indices));
    scanBlkArgs.push_back(
        b.create<memref::LoadOp>(loc, concreteOp.outputs()[0], indices));
    scanBlkArgs.push_back(
        b.create<memref::LoadOp>(loc, concreteOp.inits()[0], accIndices));
    auto &srcBlock = concreteOp.getRegion().front();
    IRMapping bvm;
    for (auto it : llvm::zip(srcBlock.getArguments(), scanBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        concreteOp.outputs()[0], indices);
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        concreteOp.inits()[0], accIndices);
    return success();
  }
};

} // namespace

template <typename OpType> static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<LinalgExtOpTilingInterface<OpType>>(*ctx);
}

void mlir::triton::linalg_ext::registerExtOpTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, triton::linalg_ext::LinalgExtDialect *dialect) {
        registerOne<triton::linalg_ext::ScatterOp>(ctx);
        registerOne<triton::linalg_ext::GatherOp>(ctx);
        registerOne<triton::linalg_ext::AtomicRMWOp>(ctx);
        registerOne<triton::linalg_ext::GatherAtomicRMWOp>(ctx);
        registerOne<triton::linalg_ext::AtomicCASOp>(ctx);
        registerOne<triton::linalg_ext::GatherAtomicCASOp>(ctx);
        registerOne<triton::linalg_ext::PadOp>(ctx);
        registerOne<triton::linalg_ext::AssertOp>(ctx);
        registerOne<triton::linalg_ext::ScanOp>(ctx);
        registerOne<triton::linalg_ext::LibdeviceCallOp>(ctx);
      });
}
