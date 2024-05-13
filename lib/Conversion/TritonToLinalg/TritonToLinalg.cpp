//===- TritonToLinalg.cpp - Triton to Linalg dialect convension -*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <map>
#include <optional>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "triton-linalg/Analysis/Utils.h"
#include "triton-linalg/Conversion/ArithToLinalg/ArithToLinalg.h"
#include "triton-linalg/Conversion/MathToLinalg/MathToLinalg.h"
#include "triton-linalg/Conversion/PassDetail.h"
#include "triton-linalg/Conversion/TritonToLinalg/AtomicCASConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/AtomicRmwConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/LoadStoreConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonToLinalg.h" // IWYU pragma: keep
#include "triton-linalg/Conversion/TritonToLinalg/TypeConverter.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/LinalgExt/Utils/Utils.h"
#include "triton-linalg/Dialect/Triton/Analysis/AxisInfoAnalysis.h"
#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Dialect/Utils/Conventions.h"
#include "triton/Common/Common.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ADL.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;
using namespace triton::dataflow;

/// Return a new shape that contains the equal elements of srcShape and
/// dstShape.
static SmallVector<int64_t, 4> findEqualDims(ArrayRef<int64_t> srcShape,
                                             ArrayRef<int64_t> dstShape) {
  assert(srcShape.size() == dstShape.size());
  SmallVector<int64_t, 4> ret;
  for (const auto [src, dst] : llvm::zip(srcShape, dstShape)) {
    if (src == dst) {
      ret.push_back(src);
    }
  }
  return ret;
}

static SmallVector<int64_t, 4>
getBroadcastDimensions(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape) {
  assert(srcShape.size() == dstShape.size());
  SmallVector<int64_t, 4> dimensions;

  for (const auto &it : llvm::enumerate(llvm::zip(srcShape, dstShape))) {
    if (std::get<0>(it.value()) != std::get<1>(it.value())) {
      assert(std::get<1>(it.value()) != 1);
      dimensions.push_back(it.index());
    }
  }
  return dimensions;
}

static Value sliceFirst(ConversionPatternRewriter &rewriter, Location loc,
                        Value input, int64_t dim) {
  ShapedType inputType = input.getType().cast<ShapedType>();
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

static Value sliceRemaining(ConversionPatternRewriter &rewriter, Location loc,
                            Value input, int64_t dim) {
  ShapedType inputType = input.getType().cast<ShapedType>();
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(inputType.getDimSize(dim) - 1);
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  offsets[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes,
                                                 strides);
}

namespace {

/// Convert an `triton.broadcast` operation to `linalg.broadcast/linalg.fill`
/// operation.
///
/// The definition of `triton.broadcast` and `linalg.broadcast` differ in the
/// following points:
///
/// 1. `triton.broadcast` support broadcast scalar to a tensor. e.g.
///
/// ```mlir
///   %1 = tt.broadcast %0 : (i32) -> tensor<128xi32>
/// ```
///
///  note: the case, matches semantic of tt.splat, has been converted to
///        tt.splat by pass of CanonicalizeTritonBroadcastPass.
///
/// 2. If operand is tensor, the rank of input operand equal to the
/// rank of output operand, it must be canonicalized. e.g.
///
/// ```mlir
///   %1 = tt.broadcast %0 : (tensor<1x128xi32>) -> tensor<32x128xi32>
/// ```
///
/// converts to:
///
/// ```mlir
///   %1 = tensor.collaspe_shape %0 [[0, 1]]
///   %2 = linalg.broadcast ins(%1: tensor<128xi32>)
///                         inits(%init: tensor<32x128xi32)
///                         dimensions = [0]
/// ```
struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();

    auto resultTy = type.cast<RankedTensorType>();
    auto loc = op.getLoc();

    // tt.broadcast with input of scalar has been converted to tt.splat,
    // no need to deal with scalar case here, just return.
    if (!op.getSrc().getType().isa<ShapedType>()) {
      return failure();
    }

    ShapedType operandTy = op.getSrc().getType().cast<ShapedType>();
    assert(operandTy.getRank() == resultTy.getRank() &&
           "rank of source and destination should match");

    auto collapseDstShape =
        findEqualDims(operandTy.getShape(), resultTy.getShape());
    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMaps(rewriter, operandTy.getShape(),
                                 collapseDstShape, reassociationMap)) {
      return rewriter.notifyMatchFailure(
          op, "tt.broadcast Attempting to expand into an incompatible shape");
    }

    // Collapse shape of src from 64x1xf32 to 64xf32.
    auto collapseSrcOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, adaptor.getSrc(), reassociationMap);

    auto initOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                   resultTy.getElementType());

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, collapseSrcOp, initOp,
        getBroadcastDimensions(operandTy.getShape(), resultTy.getShape()));

    return success();
  }
};

/// Convert an `triton.splat` operation to `linalg.fill` operation.
/// It's a special case of `tt.broadcast`.
struct TritonSplatPattern : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();
    auto resultTy = type.cast<RankedTensorType>();

    auto initOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultTy.getShape(), resultTy.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        op, ValueRange{adaptor.getSrc()}, ValueRange{initOp});
    return success();
  }
};

struct TritonExpandDimPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();
    auto resultTy = type.cast<RankedTensorType>();

    ShapedType operandTy = op.getSrc().getType().cast<ShapedType>();

    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMaps(rewriter, resultTy.getShape(),
                                 operandTy.getShape(), reassociationMap)) {
      return rewriter.notifyMatchFailure(
          op, "tt.expand_dim attempting to expand into an incompatible shape");
    }

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, resultTy, adaptor.getSrc(), reassociationMap);

    return success();
  }
};

struct TritonViewPattern : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperands()[0];
    auto operandType = operand.getType().cast<ShapedType>();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<ShapedType>();

    // Special case where the result is a 0-d tensor.
    if (resultType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, operand, ArrayRef<ReassociationIndices>{});
      return success();
    }

    // Compute the reassociation maps for the linalg operation. This will
    // succeed if the view can be done with a single expand_shape or
    // collapse_shape. e.g.
    //
    // 1. Expand operation
    //
    // ```mlir
    //   %1 = tt.reshape %0 : (tensor<256x!tt.ptr<f32>>) ->
    //   tensor<8x16x2x!tt.ptr<f32>>
    // ```
    //
    // converts to:
    //
    // ```mlir
    //   %1 = tensor.expand_shape %0 [[0, 1, 2]]
    // ```
    //
    // 2. Collapse operation
    //
    // ```mlir
    //   %1 = tt.reshape %0 : (tensor<8x16x2x!tt.ptr<f32>>) ->
    //   tensor<256x!tt.ptr<f32>>
    // ```
    //
    // converts to:
    //
    // ```mlir
    //   %1 = tensor.collapse_shape %0 [[0, 1, 2]]
    // ```
    SmallVector<ReassociationExprs, 4> reassociationMap;
    // Generate collapse operation.
    if (resultType.getRank() < operandType.getRank() &&
        createReassociationMaps(rewriter, operandType.getShape(),
                                resultType.getShape(), reassociationMap)) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, operand, reassociationMap);
      return success();
    }
    // Generate expand operation.
    if (resultType.getRank() >= operandType.getRank() &&
        createReassociationMaps(rewriter, resultType.getShape(),
                                operandType.getShape(), reassociationMap)) {
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, resultType, operand, reassociationMap);
      return success();
    }

    Value collapsedOp = operand;
    Location loc = op.getLoc();
    auto getIdentityExprs = [&rewriter](int64_t n) {
      SmallVector<AffineExpr, 4> exprs;
      for (int i = 0; i < n; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      return exprs;
    };

    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions. e.g.
    //
    // ```mlir
    //   %1 = tt.reshape %0 : (tensor<16x16x!tt.ptr<f32>>) ->
    //   tensor<4x64x!tt.ptr<f32>>
    // ```
    // converts to:
    //
    // ```mlir
    //   %1 = tensor.collapse_shape %0 [[0, 1]]
    //   %2 = tensor.expand_shape %1 [[0, 1]]
    // ```
    //
    // If there is only a single source dimension, the reduce step can be
    // skipped. TensorCollapseShape expects a different rank of operand and
    // result.
    if (operandType.getRank() != 1) {
      // Use operand_type here because we need to collapse all operands
      // dimensions.
      SmallVector<ReassociationExprs, 4> collapsingMap = {
          getIdentityExprs(operandType.getRank())};
      collapsedOp =
          rewriter.create<tensor::CollapseShapeOp>(loc, operand, collapsingMap);
    }

    if (resultType.getRank() == 1) {
      rewriter.replaceOp(op, collapsedOp);
    } else {
      // Use resultType here because we need to expand operand to
      // result's shape and type.
      SmallVector<ReassociationExprs, 4> expandingMap = {
          getIdentityExprs(resultType.getRank())};
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, resultType, collapsedOp, expandingMap);
    }
    return success();
  }
};

struct TritonAddPtrPattern : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();
    // Use diviceCeil to handle !tt.ptr<i1>.
    auto bytesPerElement =
        llvm::divideCeil(triton::getPointeeBitWidth(op.getPtr().getType()), 8);
    auto createAdd = [&rewriter, bytesPerElement](Value ptr, Value offset,
                                                  Location loc, Type type) {
      if (offset.getType() != type)
        offset = rewriter.create<arith::ExtSIOp>(loc, type, offset);
      // Since the pointer offset is by element-wise, it needs to be multiplied
      // by the byte width of the data pointed to by the pointer.
      offset = rewriter.create<arith::MulIOp>(
          loc, offset,
          rewriter.create<arith::ConstantIntOp>(loc, bytesPerElement, type));
      return rewriter.create<arith::AddIOp>(loc, ptr, offset);
    };

    Location loc = op.getLoc();
    auto resultTy = type.dyn_cast<RankedTensorType>();
    if (!resultTy) {
      // Handle addptr for scalar.
      auto ret = createAdd(adaptor.getPtr(), adaptor.getOffset(), loc, type);
      rewriter.replaceOp(op, ret->getResults());
      return success();
    }

    Value init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                  resultTy.getElementType());
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, adaptor.getOperands(), init,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto ret =
              createAdd(args[0], args[1], loc, getElementTypeOrSelf(resultTy));
          rewriter.create<linalg::YieldOp>(loc, ret->getResults());
        });
    rewriter.replaceOp(op, mapOp->getResults());

    return success();
  }
};

struct TritonMakeRangePattern
    : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();
    auto resultTy = type.cast<RankedTensorType>();

    auto initOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                   resultTy.getElementType());

    auto start = rewriter.create<arith::ConstantIntOp>(
        loc, op.getStart(), op.getStartAttr().getType());
    auto end = rewriter.create<arith::ConstantIntOp>(loc, op.getEnd(),
                                                     op.getEndAttr().getType());

    rewriter.replaceOpWithNewOp<linalg_ext::MakeRangeOp>(
        op, op.getType(), ValueRange{start, end}, ValueRange{initOp});
    return success();
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getA();
    Value filter = adaptor.getB();
    Value bias = adaptor.getC();
    auto matmulOp = rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, TypeRange{op.getType()}, ValueRange{input, filter}, bias);

    bool allowTf32 = op.getAllowTF32();
    if (allowTf32)
      matmulOp->setAttr(getAttrAllowTF32(), UnitAttr::get(op->getContext()));
    return success();
  }
};

struct TritonBitcastPattern : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type)
      return failure();
    auto resultTy = type.dyn_cast<RankedTensorType>();
    // Scalar case.
    if (!resultTy) {
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, type,
                                                    adaptor.getFrom());
      return success();
    }

    // Tensor case.
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                  resultTy.getElementType());
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, ValueRange(adaptor.getOperands()), init,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value innerResult = rewriter.create<arith::BitcastOp>(
              loc, getElementTypeOrSelf(resultTy), args);
          rewriter.create<linalg::YieldOp>(loc, innerResult);
        });
    rewriter.replaceOp(op, mapOp->getResults());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Derive types. `tt.reduce` treats reducing a 1-D tensor with a special
    // case that returns a scalar, but we treat it as a 0-D tensor in these
    // types.
    auto convertedInputTensorTypes =
        llvm::map_range(adaptor.getOperands().getTypes(),
                        [](Type t) { return t.cast<TensorType>(); });
    assert(llvm::all_equal(llvm::map_range(
        convertedInputTensorTypes, [](TensorType t) { return t.getShape(); })));
    static_cast<void>(convertedInputTensorTypes);

    auto originalResultTensorTypes =
        llvm::map_range(op.getResultTypes(), [](Type t) -> TensorType {
          if (auto tensorType = t.dyn_cast<TensorType>())
            return tensorType;
          return RankedTensorType::get({}, t);
        });
    assert(llvm::all_equal(llvm::map_range(
        originalResultTensorTypes, [](TensorType t) { return t.getShape(); })));
    ArrayRef<int64_t> resultShape =
        (*originalResultTensorTypes.begin()).getShape();
    auto convertedResultTensorTypes =
        llvm::map_range(originalResultTensorTypes, [&](TensorType t) {
          return RankedTensorType::get(resultShape, t.getElementType());
        });

    llvm::SmallVector<Value> initVals;
    // As we need to analysis the body of reduce op to get the init value,
    // currently we only support single paylod op. Otherwise, We use a portion
    // of the input as the initial value for the output.
    do {
      if (op.getNumResults() == 1) {
        Operation *payloadOp =
            linalg_ext::findPayloadOp(&op.getCombineOp().front());
        if (!payloadOp)
          break;
        std::optional<TypedAttr> fillValAttr =
            arith::getNeutralElement(payloadOp);
        // When the requirements are not met, go to the later general
        // implementation.
        if (!fillValAttr.has_value())
          break;
        Value fillVal =
            rewriter.create<arith::ConstantOp>(loc, fillValAttr.value());
        // Create empty vectors as init values.
        for (TensorType t : convertedResultTensorTypes) {
          auto initOp = rewriter.create<tensor::EmptyOp>(loc, t.getShape(),
                                                         t.getElementType());
          auto fillOp =
              rewriter.create<linalg::FillOp>(loc, fillVal, initOp.getResult());
          initVals.push_back(fillOp.getResult(0));
        }

        // Create a linalg.reduce on the same input and move the combine region
        // there. (ReduceReturnOpConversion will take care of the terminator.)
        auto reduceOp = rewriter.create<linalg::ReduceOp>(
            loc, /*resultTypes=*/SmallVector<Type>(convertedResultTensorTypes),
            /*inputs=*/adaptor.getOperands(), /*inits=*/initVals,
            /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
        rewriter.inlineRegionBefore(op.getCombineOp(), reduceOp.getCombiner(),
                                    reduceOp.getCombiner().end());

        // If the result on tt.reduce are tensors with rank > 0, we are done.
        if (!resultShape.empty()) {
          rewriter.replaceOp(op, reduceOp.getResults());
          return success();
        }

        // Otherwise, the result has to be a scalar, so we need to extract the
        // scalar from the 0-ranked result tensor.
        SmallVector<Value> results;
        Value scalar = rewriter.create<tensor::ExtractOp>(
            loc,
            SmallVector<Type>(convertedResultTensorTypes)
                .begin()
                ->dyn_cast<RankedTensorType>()
                .getElementType(),
            reduceOp->getResults()[0], /*indices=*/ValueRange{});
        results.push_back(scalar);
        rewriter.replaceOp(op, results);

        return success();
      }
    } while (false);

    llvm::SmallVector<Value> inputVals;
    // To lowering to linalg.reduce, we use the first slice of the reduction
    // axis of input operands as the init value of init operands. And then,
    // reduce the remaining elements of input operands.
    // We assume that the number of input operands is same as init operands and
    // corresponds one to one.
    // TODO: This restriction will need to be relaxed in the future.
    assert(adaptor.getOperands().size() == op.getNumResults() &&
           "tt.reduce requires the same input number and init number");
    for (auto [inputVal, initTy] :
         llvm::zip(adaptor.getOperands(), convertedResultTensorTypes)) {
      ShapedType inputTy = inputVal.getType().cast<ShapedType>();
      ArrayRef<int64_t> inputShape = inputTy.getShape();

      // If the size of reduce axis is 1, we will replace init operands by input
      // operands, so we should resize the input operands' shape by init
      // operands.
      if (inputShape[op.getAxis()] <= 1) {
        assert(inputVals.empty() &&
               "tt.reduce requires the same shape of all input operands");
        SmallVector<ReassociationExprs, 4> reassociationMap;
        [[maybe_unused]] bool res = createReassociationMaps(
            rewriter, inputShape, initTy.getShape(), reassociationMap);
        assert(res && "attempting to collapse into an incompatible shape");
        auto collapse = rewriter.create<tensor::CollapseShapeOp>(
            loc, inputVal, reassociationMap);
        initVals.push_back(collapse);
        continue;
      }

      // 1. Slice the first elements of input operands, and use them as init
      //    operands' init value.
      {
        Value slice = sliceFirst(rewriter, loc, inputVal, op.getAxis());
        auto sliceShape = slice.getType().cast<ShapedType>().getShape();

        // Resize slice value's shape by init operand.
        SmallVector<ReassociationExprs, 4> reassociationMap;
        [[maybe_unused]] bool res = createReassociationMaps(
            rewriter, sliceShape, initTy.getShape(), reassociationMap);
        assert(res && "attempting to collapse into an incompatible shape");
        auto collapse = rewriter.create<tensor::CollapseShapeOp>(
            loc, slice, reassociationMap);
        initVals.push_back(collapse);
      }

      // 2. Slice the remaining elements of input operands, reduce them and
      //    init value.
      {
        Value slice = sliceRemaining(rewriter, loc, inputVal, op.getAxis());
        inputVals.push_back(slice);
      }
    }

    // If the results are scalar, we need to extract the scalar from the
    // 0-ranked result tensor.
    auto getFinalResults = [&](ValueRange results) -> SmallVector<Value> {
      if (!resultShape.empty())
        return results;
      SmallVector<Value> extractResults;
      for (auto [tensor, type] :
           llvm::zip(results, convertedResultTensorTypes)) {
        Value scalar = rewriter.create<tensor::ExtractOp>(
            loc, type.getElementType(), tensor, /*indices=*/ValueRange{});
        extractResults.push_back(scalar);
      }
      return extractResults;
    };

    // If the the size of reduce axis is 1, we just replace the init operands by
    // input operands.
    if (inputVals.empty()) {
      rewriter.replaceOp(op, getFinalResults(initVals));
      return success();
    }

    // Create a linalg.reduce on the same input and move the combine region
    // there. (ReduceReturnOpConversion will take care of the terminator.)
    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, /*resultTypes=*/SmallVector<Type>(convertedResultTensorTypes),
        /*inputs=*/inputVals, /*inits=*/initVals,
        /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
    rewriter.inlineRegionBefore(op.getCombineOp(), reduceOp.getCombiner(),
                                reduceOp.getCombiner().end());

    rewriter.replaceOp(op, getFinalResults(reduceOp.getResults()));
    return success();
  }
};

struct TritonReduceReturnPattern
    : public OpConversionPattern<triton::ReduceReturnOp> {
  using OpConversionPattern<triton::ReduceReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct TritonPtrToIntPattern : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern<triton::PtrToIntOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct TritonIntToPtrPattern : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern<triton::IntToPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

struct OptimizationBarrierOpPattern
    : public OpConversionPattern<aux::OptimizationBarrierOp> {
  using OpConversionPattern<aux::OptimizationBarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aux::OptimizationBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<aux::OptimizationBarrierOp>(
        op, adaptor.getOperand());
    return success();
  }
};

class PtrSelectOpPattern : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

public:
  PtrSelectOpPattern(TritonLinalgTypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<arith::SelectOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getOperands());
    return success();
  }
};

struct GPUBarrierOpPattern : public OpConversionPattern<gpu::BarrierOp> {
  using OpConversionPattern<gpu::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType srcTy =
        adaptor.getSrc().getType().cast<RankedTensorType>();
    auto rank = srcTy.getRank();
    if (rank <= 1) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    SmallVector<int64_t> permutation =
        llvm::to_vector(llvm::reverse(llvm::seq<int64_t>(0, rank)));
    SmallVector<int64_t> retShape(srcTy.getShape().rbegin(),
                                  srcTy.getShape().rend());
    auto initOp =
        rewriter.create<tensor::EmptyOp>(loc, retShape, srcTy.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op, adaptor.getSrc(),
                                                     initOp, permutation);

    return success();
  }
};

class TritonReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

// Copied from mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp.
class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the original function results.
    SmallVector<Type, 1> convertedResults;
    if (failed(typeConverter->convertTypes(callOp.getResultTypes(),
                                           convertedResults)))
      return failure();

    // If this isn't a one-to-one type mapping, we don't know how to aggregate
    // the results.
    if (callOp->getNumResults() != convertedResults.size())
      return failure();

    // Substitute with the new result types from the corresponding FuncType
    // conversion.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        callOp, callOp.getCallee(), convertedResults, adaptor.getOperands());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getFunctionType().cast<FunctionType>();
    auto *converter = getTypeConverter();
    // Convert the original function types.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(converter->convertSignatureArgs(type.getInputs(), result)) ||
        failed(converter->convertTypes(type.getResults(), newResults)))
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     result.getConvertedTypes(), newResults);
    auto newOp =
        rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newType);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Special handle arith::cmpi that it just compute the mask.
/// Change it to linalg.fill.
class ArithCmpIToFillPattern : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaskTracker tracker;
    auto loc = op.getLoc();
    tracker.parse(op.getResult(), loc, rewriter);
    if (tracker.hasFailedDim())
      return failure();

    auto tensorType =
        op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    if (!tensorType)
      return failure();

    Value init = rewriter.create<tensor::EmptyOp>(loc, tracker.getSizes(),
                                                  tensorType.getElementType());
    Value trueVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    Value one =
        rewriter.create<linalg::FillOp>(loc, trueVal, init).getResult(0);

    Value falseVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));

    SmallVector<OpFoldResult> offsets = llvm::to_vector(tracker.getStarts());
    // Replace with pad op.
    auto value = getPadOrInsertOpWithOther(
        loc, falseVal, tensorType, one, offsets, tracker.getSizes(), rewriter);
    rewriter.replaceOp(op, value);
    return success();
  }
};

/// Special handle arith::select that it just select true data by the mask.
/// Change it to insert true data to false data.
class ArithSelectConversionPattern
    : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto cond = op.getCondition();
    Value trueValue = op.getTrueValue();
    Value falseValue = op.getFalseValue();
    if (!cond.getType().dyn_cast_or_null<ShapedType>())
      return failure();

    MaskTracker tracker;
    tracker.parse(cond, loc, rewriter);
    // Check if the cond is from arith.andi.
    if (tracker.hasFailedDim()) {
      auto preOp = cond.getDefiningOp<arith::AndIOp>();
      if (!preOp)
        return failure();
      // Check if arith.andi left operand has mask state (Note: arith.andi has
      // been canonicalized in createCanonicalizeTritonPass).
      MaskTracker operandTracker;
      operandTracker.parse(preOp.getLhs(), loc, rewriter);
      if (operandTracker.hasFailedDim())
        return failure();
      Value nonMask = preOp.getRhs();
      trueValue = rewriter.create<arith::SelectOp>(
          loc, nonMask, adaptor.getTrueValue(), adaptor.getFalseValue());
      // Update mask state.
      tracker = operandTracker;
    }

    auto srcType = trueValue.getType().dyn_cast_or_null<RankedTensorType>();
    if (!srcType)
      return failure();
    auto rank = srcType.getRank();
    SmallVector<OpFoldResult> offsets = llvm::to_vector(tracker.getStarts());
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    auto dstType = tensor::ExtractSliceOp::inferResultType(
        srcType, offsets, tracker.getSizes(), strides);
    auto srcSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, dstType, trueValue, offsets, tracker.getSizes(), strides);
    // Replace with pad or insert_slice op.
    auto value = getPadOrInsertOpWithOther(
        loc, falseValue, op.getResult().getType(), srcSlice, offsets,
        tracker.getSizes(), rewriter);
    rewriter.replaceOp(op, value);
    return success();
  }
};

class TritonPrintPattern : public OpConversionPattern<triton::PrintOp> {
public:
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int numOperands = op.getNumOperands();

    /// Print pid information.
    SmallVector<Value, 3> pid;
    for (int axis = 0; axis < 3; axis++) {
      auto getProgramIdOp = rewriter.create<triton::GetProgramIdOp>(
          loc, rewriter.getI32Type(),
          triton::ProgramIDDimAttr::get(rewriter.getContext(),
                                        triton::ProgramIDDim(axis)));
      pid.push_back(getProgramIdOp.getResult());
    }
    assert(pid.size() == 3 && "Expected size of pid to be 3");
    SmallVector<StringAttr, 4> pidPrefix{
        rewriter.getStringAttr("pid ("), rewriter.getStringAttr(", "),
        rewriter.getStringAttr(", "), rewriter.getStringAttr(") ")};
    for (size_t i = 0; i < 3; i++) {
      rewriter.create<aux::ScalarPrintOp>(loc, pid[i], pidPrefix[i]);
    }
    rewriter.create<aux::ScalarPrintOp>(loc, nullptr, pidPrefix[3]);

    // Split tt.print based on whether it is a scalar or tensor:
    // 1. If op.getPrefix() is not empty, print op.getPrefix() using
    // aux.scalar.print.
    // 2. If all inputs are scalars, replace tt.print with aux.scalar.print. If
    // there are multiple inputs, split into multiple aux.scalar.print.
    // 3. If all inputs are tensors, replace tt.print with aux.print. If there
    // are multiple inputs, split into multiple aux.print.
    // 4. If there are both scalars and tensors, split tt.print into
    // aux.scalar.print and aux.print.
    StringAttr prefix = op.getPrefixAttr();
    if (numOperands == 0) {
      rewriter.create<aux::ScalarPrintOp>(loc, nullptr, prefix);
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 0; i < numOperands; ++i) {
      auto oriOperandType = getElementTypeOrSelf(op.getOperands()[i].getType());
      if (oriOperandType.isa<triton::PointerType>()) {
        prefix = rewriter.getStringAttr(op.getPrefix() + Twine("%p"));
      }
      auto operand = adaptor.getOperands()[i];
      auto operandType = typeConverter->convertType(operand.getType());
      if (!operandType)
        return failure();
      auto resultTy = operandType.dyn_cast<RankedTensorType>();

      if (!resultTy) {
        rewriter.create<aux::ScalarPrintOp>(loc, operand, prefix);
      } else {
        auto printOp = rewriter.create<aux::PrintOp>(loc, resultTy, operand,
                                                     operand, prefix);
        DominanceInfo dom(op->getParentOp());
        operand.replaceUsesWithIf(printOp.getResult(),
                                  [&](OpOperand &op) -> bool {
                                    Operation *user = op.getOwner();
                                    return dom.properlyDominates(printOp, user);
                                  });
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class TritonAssertOpPattern : public OpConversionPattern<triton::AssertOp> {
public:
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condVal = op.getCondition();
    auto valType = condVal.getType();

    auto assertMessage =
        llvm::formatv("{0}:{1}: {2} Assertion `{3}` failed", op.getFile(),
                      op.getLine(), op.getFunc(), op.getMessage());
    auto resultTy = valType.cast<RankedTensorType>();

    // Only supports int type.
    // follow:
    // http://gitlab.software.cambricon.com/neuware/triton/-/blob/main-llvm-17/lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp#L268
    assert(resultTy.getElementType().isa<mlir::IntegerType>() &&
           "Only support int tensor for assert");

    rewriter.create<linalg_ext::AssertOp>(op.getLoc(), resultTy, condVal,
                                          assertMessage.str());

    rewriter.eraseOp(op);
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // If the the size of scan axis is 1, we just replace op by
    // input operands.
    if (adaptor.getOperands()[0]
            .getType()
            .cast<ShapedType>()
            .getShape()[op.getAxis()] <= 1) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    auto convertedInputTensorTypes =
        llvm::map_range(adaptor.getOperands().getTypes(),
                        [](Type t) { return t.cast<TensorType>(); });
    assert(llvm::all_equal(llvm::map_range(
        convertedInputTensorTypes, [](TensorType t) { return t.getShape(); })));
    static_cast<void>(convertedInputTensorTypes);

    llvm::SmallVector<Value> inputVals;
    llvm::SmallVector<Value> initVals;
    assert(adaptor.getOperands().size() == op.getNumResults() &&
           "tt.scan requires the same input number and init number");
    assert(adaptor.getOperands().size() == 1 &&
           "tt.scan only support single input now");
    for (auto inputVal : adaptor.getOperands()) {
      RankedTensorType inputTy = inputVal.getType().cast<RankedTensorType>();
      int64_t rank = inputTy.getRank();

      // 1. Slice the remaining elements of input operands.
      {
        Value slice = sliceRemaining(rewriter, loc, inputVal, op.getAxis());
        // Create output tensor
        auto sliceShape = slice.getType().cast<ShapedType>().getShape();
        Value empty = rewriter.create<tensor::EmptyOp>(
            loc, sliceShape, inputTy.getElementType());
        inputVals.push_back(slice);
        initVals.push_back(empty);
      }

      // 2. Slice the first elements of input operands, and use them as init
      //    operands' init value.
      {
        Value slice = sliceFirst(rewriter, loc, inputVal, op.getAxis());
        // Create the reshape of slice tensor.
        SmallVector<Value> reshapeSizes;
        SmallVector<int64_t> reshapeShape;
        for (int64_t i = 0; i < rank; ++i) {
          if (i != op.getAxis()) {
            reshapeSizes.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, inputTy.getShape()[i]));
            reshapeShape.push_back(inputTy.getShape()[i]);
          }
        }
        auto reshapeType =
            RankedTensorType::get(reshapeShape, inputTy.getElementType());
        auto reshapeSizesTensor = rewriter.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get(rank - 1, rewriter.getIndexType()),
            ValueRange{reshapeSizes});
        Value reshape = rewriter.create<tensor::ReshapeOp>(
            loc, reshapeType, slice, reshapeSizesTensor);
        initVals.push_back(reshape);
      }
    }

    // Create a linalg_ext.scan on the same input and move the combine region
    // there. (ScanReturnOpConversion will take care of the terminator.)
    auto resultTypes = llvm::map_range(
        initVals, [](Value t) { return t.getType().cast<RankedTensorType>(); });

    auto scanOp = rewriter.create<linalg_ext::ScanOp>(
        loc, /*resultTypes=*/SmallVector<Type>(resultTypes),
        /*inputs=*/inputVals, /*inits=*/initVals,
        /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
    auto &block = op->getRegion(0).front();
    block.addArgument(block.getArgumentTypes()[0], loc);
    rewriter.replaceAllUsesWith(block.getArgument(1), block.getArgument(2));
    rewriter.inlineRegionBefore(op.getCombineOp(), scanOp.getCombiner(),
                                scanOp.getCombiner().end());

    // Insert linalg_ext.scan result into input operand.
    // Retrieve insert sizes of result tensor.
    RankedTensorType initType = initVals[0].getType().cast<RankedTensorType>();
    ArrayRef<int64_t> initShape = initType.getShape();
    int64_t rank = initType.getRank();
    auto insertSizes =
        llvm::to_vector(llvm::map_range(initShape, [&](int64_t t) {
          return OpFoldResult(rewriter.getI64IntegerAttr(t));
        }));

    // Retrieve insert offsets of result tensor.
    SmallVector<OpFoldResult> insertOffsets(rank, rewriter.getIndexAttr(0));
    insertOffsets[op.getAxis()] = rewriter.getIndexAttr(1);

    // Retrieve insert strides of result tensor.
    SmallVector<OpFoldResult> insertStrides(rank, rewriter.getIndexAttr(1));

    auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, scanOp.getResults()[0], adaptor.getOperands()[0], insertOffsets,
        insertSizes, insertStrides);
    rewriter.replaceOp(op, insertSliceOp.getResult());
    return success();
  }
};

struct TritonScanReturnPattern
    : public OpConversionPattern<triton::ScanReturnOp> {
  using OpConversionPattern<triton::ScanReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> doubleOperands;
    for (int64_t i = 0; i < 2; ++i) {
      for (auto operand : adaptor.getOperands())
        doubleOperands.push_back(operand);
    }
    rewriter.replaceOpWithNewOp<linalg_ext::ExtYieldOp>(op, doubleOperands);
    return success();
  }
};

} // namespace

static void
populateTritonToLinalgPatterns(RewritePatternSet &patterns,
                               TritonLinalgTypeConverter &converter) {
  MLIRContext *context = patterns.getContext();
  patterns
      .add<TritonBroadcastPattern, TritonSplatPattern, TritonExpandDimPattern,
           TritonAddPtrPattern, TritonMakeRangePattern, TritonDotPattern,
           TritonBitcastPattern, TritonReducePattern, TritonReduceReturnPattern,
           TritonPtrToIntPattern,
           TritonIntToPtrPattern, TritonTransPattern, TritonReturnOpConversion,
           TritonCallOpPattern, TritonFuncOpPattern, TritonViewPattern,
           TritonPrintPattern, TritonAssertOpPattern, TritonScanPattern,
           TritonScanReturnPattern>(converter, context);
}

static void populateArithConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ArithCmpIToFillPattern, ArithSelectConversionPattern>(context);
}

namespace {

struct TritonToLinalgPass : public TritonToLinalgPassBase<TritonToLinalgPass> {
  void runOnOperation() override;
};
} // namespace

void TritonToLinalgPass::runOnOperation() {
  MLIRContext *context = &getContext();
  TritonLinalgTypeConverter converter;
  ConversionTarget target(*context);
  target.addLegalDialect<BuiltinDialect, func::FuncDialect>();
  target.addIllegalDialect<triton::TritonDialect, gpu::GPUDialect>();
  // Mark MakeTensorOp and AdvanceOp as legal op, it will be erased by cse.
  target.addDynamicallyLegalOp<triton::MakeTensorPtrOp, triton::AdvanceOp>(
      [](Operation *op) { return !op->getUsers().empty(); });
  target
      .addLegalOp<LLVM::IntToPtrOp, triton::aux::StoreResourceOp, aux::ViewOp,
                  bufferization::ToTensorOp, bufferization::ToMemrefOp,
                  bufferization::MaterializeInDestinationOp, aux::PrintOp,
                  aux::ScalarPrintOp>();
  target.addDynamicallyLegalDialect<
      tensor::TensorDialect, linalg::LinalgDialect,
      triton::linalg_ext::LinalgExtDialect, scf::SCFDialect>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      [&](Operation *op) {
        return !op->getResultTypes().front().isa<ShapedType>();
      });
  target.addDynamicallyLegalOp<aux::OptimizationBarrierOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  target.addDynamicallyLegalOp<arith::SelectOp>([&](Operation *op) {
    auto resType = op->getResultTypes().front();
    return !resType.isa<ShapedType>() && converter.isLegal(op);
  });
  target.addLegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();

  auto solver = createDataFlowSolver();
  solver->load<AxisInfoAnalysis>();
  if (failed(solver->initializeAndRun(getOperation())))
    return signalPassFailure();

  // Rewrite patterns.
  RewritePatternSet patterns(context);
  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  patterns.add<OptimizationBarrierOpPattern, GPUBarrierOpPattern>(converter,
                                                                  context);
  patterns.add<PtrSelectOpPattern>(converter, context, 0);
  populateTritonToLinalgPatterns(patterns, converter);
  populateArithToLinalgPatterns(patterns);
  populateArithConversionPatterns(patterns);
  populateMathToLinalgPatterns(patterns);

  populateTritonLoadStoreToLinalgPatterns(patterns, converter, *solver);
  populateTritonAtomicCASToLinalgPatterns(patterns, converter, *solver);
  populateTritonAtomicRmwToLinalgPatterns(patterns, converter, *solver);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
