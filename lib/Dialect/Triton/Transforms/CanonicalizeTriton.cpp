//===- CanonicalizeTriton.cpp -triton ir canonicalization pass---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <memory>
#include <optional>
#include <queue>
#include <stddef.h>
#include <stdint.h>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Triton/Transforms/PassDetail.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerInfo.h"
#include "triton-linalg/Dialect/Utils/ArithUtils.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace triton;

namespace mlir {
class MLIRContext;
} // namespace mlir

static bool isAllOneSizeType(ShapedType inputType) {
  return inputType.getRank() == 0 ||
         llvm::all_of(inputType.getShape(),
                      [](int64_t size) { return size == int64_t(1); });
}

/// Extract cond value from mask.
static Value extractCondFromShapeMask(Value mask, PatternRewriter &rewriter) {
  if (!mask)
    return mask;
  auto loc = mask.getLoc();
  if (auto maskValType = dyn_cast<ShapedType>(mask.getType())) {
    assert(isAllOneSizeType(maskValType) ||
           isa_and_nonnull<arith::ConstantOp>(mask.getDefiningOp()));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices;
    for (int64_t i = 0; i < maskValType.getRank(); ++i)
      indices.push_back(zero);
    return rewriter.create<tensor::ExtractOp>(loc, mask, ValueRange(indices));
  }
  return mask;
}

/// Get cond value from mask.
static Value getCondVal(Value mask, PatternRewriter &rewriter) {
  if (!mask)
    return nullptr;

  // If the mask value is a 0-rank tensor or a tensor with all dimensions of
  // one-size, return it.
  // For example:
  // - %mask : tensor<i1>
  // - %mask = tensor<1x1xi1>
  if (auto maskTy = dyn_cast<ShapedType>(mask.getType())) {
    if (isAllOneSizeType(maskTy))
      return extractCondFromShapeMask(mask, rewriter);
  }

  // The origin mask must be obtained by broadcasting an i1.
  auto defineOp = mask.getDefiningOp();
  Value scalarMask = nullptr;
  if (defineOp) {
    llvm::TypeSwitch<Operation *>(defineOp)
        // %mask = tt.splat i1 -> <axbxi1>
        .Case<triton::SplatOp>(
            [&](triton::SplatOp splatOp) { scalarMask = splatOp.getSrc(); })
        // %mask = tt.broadcast tensor<i1> -> <axbxi1>
        // %mask = tt.broadcast tensor<1x1xi1> -> <ax1xbx1xi1>
        .Case<triton::BroadcastOp>([&](triton::BroadcastOp broadcastOp) {
          Value src = broadcastOp.getSrc();
          if (isAllOneSizeType(cast<ShapedType>(src.getType()))) {
            scalarMask = src;
          } else {
            scalarMask = getCondVal(src, rewriter);
          }
        })
        // %mask = arith.constant dense : tensor<axbxi1>
        .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
          auto value = dyn_cast<DenseElementsAttr>(constantOp.getValue());
          if (value && value.isSplat())
            scalarMask = constantOp.getResult();
        });
  }
  return extractCondFromShapeMask(scalarMask, rewriter);
}

/// Check if any dimension's size equals to one and skip to coalesce if any
/// coalescable dimension needs boundary checking.
template <typename OpTy>
static SmallVector<int64_t>
getDegeneratableDimensions(OpTy op, const ArrayRef<int64_t> degeneratedDims) {
  RankedTensorType tensorType;
  auto loadOp = dyn_cast<triton::LoadOp>(*op);
  auto storeOp = dyn_cast<triton::StoreOp>(*op);
  if (loadOp) {
    tensorType =
        dyn_cast_if_present<RankedTensorType>(loadOp.getResult().getType());
  } else if (storeOp) {
    tensorType =
        dyn_cast_if_present<RankedTensorType>(storeOp.getValue().getType());
  } else {
    return {};
  }
  if (!tensorType || degeneratedDims.empty())
    return {};

  SmallVector<int64_t> degeneratableDims;
  std::optional<ArrayRef<int32_t>> boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck || boundaryCheck->empty()) {
    degeneratableDims.append(degeneratedDims.begin(), degeneratedDims.end());
  } else {
    for (auto dim : degeneratedDims) {
      if (llvm::find(boundaryCheck.value(), dim) ==
          boundaryCheck.value().end()) {
        degeneratableDims.push_back(dim);
      }
    }
  }

  return degeneratableDims;
}

/// Calculate degenerated boundary check by the degeneratable dimensions.
template <typename OpTy>
static SmallVector<int32_t>
getDegeneratedBoundaryCheck(OpTy op,
                            const ArrayRef<int64_t> degeneratableDims) {
  SmallVector<int32_t> degeneratedBoundaryCheck(op.getBoundaryCheck());
  for (int64_t dim : llvm::reverse(degeneratableDims)) {
    llvm::for_each(llvm::enumerate(degeneratedBoundaryCheck),
                   [&dim, &degeneratedBoundaryCheck](auto bc) {
                     if (bc.value() > dim)
                       --degeneratedBoundaryCheck[bc.index()];
                   });
  }
  return degeneratedBoundaryCheck;
}

/// Donate the MaskInfo extract from mask.
struct MaskInfo {
  enum LogicalType { AND, OR, NONE };
  Value cond;              ///< Mask which is splat or broadcast by i1 scalar.
  Value mask;              ///< Tensor or i1, other mask.
  LogicalType logicalType; ///< The logical type of cond and mask.
};

/// Get mask info mask which separate mask from splat scalar i1 or broadcast by
/// tensor<1xi1> with others.
///
/// Example:
///
/// ```mlir
/// %mask0 = ...
/// %mask1 = tt.splat %scalar
/// %mask2 = tt.broadcast %tensor0d
/// %mask01 = arith.andi %mask0, %mask1
/// %mask = arith.andi %mask01, %mask2
/// ```
///
/// Given %mask, returns %splatMask and %otherMask.
///
/// ```mlir
/// %extract = tensor.extract %tensor0d
/// %cond = arith.andi %scalar, %extract
/// %mask = %mask0
/// ```
MaskInfo getMaskInfo(Value mask, PatternRewriter &rewriter) {
  MaskInfo maskInfo = {nullptr, nullptr, MaskInfo::LogicalType::NONE};
  if (!mask)
    return maskInfo;

  auto loc = mask.getLoc();

  using LogicalType = MaskInfo::LogicalType;
  auto createMasks = [&rewriter, loc](ArrayRef<Value> masks,
                                      LogicalType type) -> Value {
    if (masks.empty())
      return nullptr;
    auto size = masks.size();
    Value ret = masks[0];
    for (size_t i = 1; i < size; i++) {
      if (type == LogicalType::AND) {
        ret = rewriter.create<arith::AndIOp>(loc, ret, masks[i]);
      } else {
        assert(type == LogicalType::OR);
        ret = rewriter.create<arith::OrIOp>(loc, ret, masks[i]);
      }
    }

    return ret;
  };

  SmallVector<Value> condMasks, otherMasks;
  std::queue<Value> worklist;
  worklist.push(mask);
  LogicalType logicalType = LogicalType::NONE;
  while (!worklist.empty()) {
    auto item = worklist.front();
    worklist.pop();
    auto defOp = item.getDefiningOp();
    if (llvm::isa_and_nonnull<arith::AndIOp, arith::OrIOp>(defOp)) {
      worklist.push(defOp->getOperand(0));
      worklist.push(defOp->getOperand(1));
      LogicalType currentLogicalType =
          llvm::isa<arith::AndIOp>(defOp) ? LogicalType::AND : LogicalType::OR;
      if (logicalType == LogicalType::NONE) {
        logicalType = currentLogicalType;
      } else if (logicalType != currentLogicalType) {
        // TODO: add support for mixed logical type.
        return maskInfo;
      }
      continue;
    }

    Value cond = getCondVal(item, rewriter);
    if (cond) {
      condMasks.push_back(cond);
      continue;
    }

    otherMasks.push_back(item);
  }
  if (condMasks.empty())
    return maskInfo;

  // Only has cond, set logical type to AND.
  if (otherMasks.empty())
    logicalType = LogicalType::AND;

  maskInfo.logicalType = logicalType;
  maskInfo.cond = createMasks(condMasks, logicalType);
  maskInfo.mask = createMasks(otherMasks, logicalType);

  return maskInfo;
}

namespace {
/// MaskOpsOrganizer is a template class designed for reorganizing
/// operations of type OpTy chain based on certain masking conditions.
/// It categorizes the operations into masked and non-masked groups.
///
/// Example:
/// mask non-mask                           non-mask non-mask
///   \  /             is transformed into        \  /
///    op   non-mask  ====================>  mask  op
///     \  /                                    \  /
///      op                                      op
///
template <typename OpTy> class MaskOpsOrganizer {
public:
  MaskOpsOrganizer(OpTy op) : srcOp(op) {}
  // Reorganize the source operation (srcOp) chain by dividing it into
  // masked and non-masked branches.
  void reorganize();

private:
  // Divide the branches of the given operation into masked and non-masked
  // groups recursively.
  void divideBranches(OpTy op);

private:
  OpTy srcOp;
  llvm::SmallVector<Value> maskVals;
  llvm::SmallVector<Value> nonMaskVals;
};

template <typename OpTy> void MaskOpsOrganizer<OpTy>::divideBranches(OpTy op) {
  MaskTracker tracker;
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  tracker.parse(op.getResult(), op.getLoc(), rewriter);
  if (!tracker.hasFailedDim()) {
    maskVals.push_back(op.getResult());
    return;
  }
  // Traverse op inputs.
  for (auto input : op->getOperands()) {
    auto preOp = input.getDefiningOp();
    if (!preOp) {
      continue;
    }
    MaskTracker operandTracker;
    operandTracker.parse(input, op.getLoc(), rewriter);
    if (!operandTracker.hasFailedDim()) {
      maskVals.push_back(input);
      continue;
    }
    // If pre op is OpTy op, call recursively.
    if (isa<OpTy>(preOp)) {
      divideBranches(llvm::cast<OpTy>(preOp));
    } else {
      nonMaskVals.push_back(input);
    }
  }
}

template <typename OpTy> void MaskOpsOrganizer<OpTy>::reorganize() {
  // Divide branches to mask branches and non-mask branches.
  divideBranches(srcOp);

  if (maskVals.empty() || nonMaskVals.empty()) {
    return;
  }
  // Construct mask branches.
  IRRewriter rewriter(srcOp.getContext());
  rewriter.setInsertionPoint(srcOp);
  Value nextMaskInput;
  for (auto [index, curVal] : llvm::enumerate(maskVals)) {
    if (index == 0) {
      nextMaskInput = curVal;
      continue;
    }
    nextMaskInput =
        rewriter.create<OpTy>(srcOp.getLoc(), nextMaskInput, curVal);
  }
  // Construct non-mask branches.
  Value nextNonMaskInput;
  for (auto [index, curVal] : llvm::enumerate(nonMaskVals)) {
    if (index == 0) {
      nextNonMaskInput = curVal;
      continue;
    }
    nextNonMaskInput =
        rewriter.create<OpTy>(srcOp.getLoc(), nextNonMaskInput, curVal);
  }
  // Connect mask and non-mask branches and replace.
  rewriter.replaceOpWithNewOp<OpTy>(srcOp, nextMaskInput, nextNonMaskInput);
}

/// Triton broadcast support operation on three cases, say, (1) expand a scalar
/// or 0d tensor to a tensor with higher rank, (2) expand a tensor (except for
/// the 0d tensor) to one with higher rank,  (3) expand on the one axis out of
/// input's axes if the rank of input tensor equals to that of output tensor.
/// Case 3 is the standard use case of tt.broadcast, we are going to
/// canonicalize the first two cases to the standard case.
///
/// Specifically,
/// for case 1, we convert the tt.broadcast operation to tt.splat since the
/// the case matches semantic of tt.splat;
/// for case 2, we use tt.expand_dims expand the rank of input tensor to be
/// same as that of output tensor at first, then, perform tt.broadcast
/// operation on the expanded tensor.
/// for case 3, do nothing since the case is the standard use case of broadcast.
///
/// Example:
///
/// For case 1, convert the tt.broadcast operation on
/// ``` mlir
///   tt.broadcast %0 : (i32) -> tensor<8x2xi32>
/// ```
/// to
/// ``` mlir
///   tt.splat %0 : (i32) -> tensor<8x2xi32>
/// ```
/// For case 2, convert the tt.broadcast operation on
/// ``` mlir
///   tt.broadcast %arg0 : (tensor<8xf32>) -> tensor<2x8xf32>
/// ```
/// to
/// ``` mlir
///   %0 = tt.expand_dims %arg0 {axis = 0 : i32} : (tensor<8xf32>) ->
///   tensor<1x8xf32>
///   %1 = tt.broadcast %0 : (tensor<1x8xf32>) ->
///   tensor<2x8xf32>
/// ```
/// For case 3, do nothing for tt.broadcast operation on
/// ``` mlir
///   tt.broadcast %arg0 : (tensor<2x1x8xf32>) -> tensor<2x2x8xf32>
/// ```
class CanonicalizeTtBroadCastPattern
    : public OpRewritePattern<triton::BroadcastOp> {
public:
  explicit CanonicalizeTtBroadCastPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::BroadcastOp>(ctx) {}

  LogicalResult matchAndRewrite(triton::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    if (isScalarOr0dTensor(op.getSrc())) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getType(),
                                                   op.getSrc());
      return success();
    }

    ShapedType inputShape = cast<ShapedType>(op.getSrc().getType());
    ShapedType outputShape = cast<ShapedType>(op.getResult().getType());
    // Meet case 3 described above, so do nothing but return.
    if (inputShape.getRank() == outputShape.getRank()) {
      return failure();
    }
    ArrayRef<int64_t> inputShapeArr = inputShape.getShape();
    ArrayRef<int64_t> outputShapeArr = outputShape.getShape();
    int numOfDimsToBeExpanded =
        getNumberOfDimsToBeExpanded(inputShapeArr, outputShapeArr);
    Value expanded = op.getSrc();
    for (int i = 0; i < numOfDimsToBeExpanded; ++i) {
      // Always pad on the leftmost side.
      expanded =
          rewriter.create<triton::ExpandDimsOp>(op.getLoc(), expanded, 0);
    }
    if (llvm::equal(cast<ShapedType>(expanded.getType()).getShape(),
                    outputShapeArr)) {
      rewriter.replaceOp(op, expanded);
    } else {
      rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
          op, op.getResult().getType(), expanded);
    }
    return success();
  }

private:
  bool isScalarOr0dTensor(const Value &val) const {
    return !isa<ShapedType>(val.getType()) ||
           cast<ShapedType>(val.getType()).getRank() == 0;
  }

  /// Returns how many axes should be expanded to input
  int getNumberOfDimsToBeExpanded(const ArrayRef<int64_t> inputShape,
                                  const ArrayRef<int64_t> outputShape) const {
    const int inputShapeSize = inputShape.size();
    const int outputShapeSize = outputShape.size();
    const int outNum = outputShapeSize - inputShapeSize;
    return outNum;
  }
};

/// This pattern complements a default other field or padding option with zero
/// value. For `tt.load` without other field and pad option, the behaviour of
/// `convert-triton-gpu-to-llvm` is atomically reset the unmasked region to
/// zero. To keep consistent, we explicitly add the `PAD_ZERO` option for tensor
/// pointer type with boundary check semantics, or add other field with zero
/// value for other cases with mask semantics. Example 1:
///
/// ``` mlir
///   %0 = tt.load %ptr {boundaryCheck = array<i32: 0>, cache = 1 : i32,
///                      evict = 1 : i32, isVolatile = false}
///        : !tt.ptr<tensor<8xf32>> -> tensor<8xf32>
/// ```
/// is converted to:
///
/// ``` mlir
///   %0 = tt.load %ptr {boundaryCheck = array<i32: 0>, cache = 1 : i32,
///                      evict = 1 : i32, isVolatile = false, padding = 1 : i32}
///        : !tt.ptr<tensor<8xf32>> -> tensor<8xf32>
/// ```
///
/// Example 2:
///
/// ``` mlir
///   %0 = tt.load %ptr, %mask
///        {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
/// ```
/// is converted to:
///
/// ``` mlir
///   %cst = arith.constant 0 : f32
///   %0 = tt.load %ptr, %mask, %cst
///        {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
/// ```
///
/// Example 3:
///
/// ``` mlir
///   %0 = tt.load %ptr, %mask {cache = 1 : i32, evict = 1 : i32,
///                             isVolatile = false}
///        : tensor<8xbf16>
/// ```
/// is converted to:
///
/// ``` mlir
///   %cst = arith.constant dense<0.000000e+00> : tensor<8xbf16>
///   %0 = tt.load %ptr, %mask, %cst {cache = 1 : i32, evict = 1 : i32,
///                                   isVolatile = false}
///        : tensor<8xbf16>
/// ```
class CanonicalizeTtLoadPattern : public OpRewritePattern<triton::LoadOp> {
public:
  explicit CanonicalizeTtLoadPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::LoadOp>(ctx) {}

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ptrType = op.getPtr().getType();
    // For op with tnesor pointer type, append `PAD_ZERO` as the padding option.
    if (triton::isTensorPointerType(ptrType)) {
      std::optional<ArrayRef<int32_t>> boundaryCheck = op.getBoundaryCheck();
      // Only append attribute when boundary check is not empty and padding
      // option attribute does not exist.
      if (!boundaryCheck || boundaryCheck->empty() || op.getPadding())
        return failure();

      auto defaultAttr = triton::PaddingOptionAttr::get(
          op.getContext(), triton::PaddingOption::PAD_ZERO);
      rewriter.modifyOpInPlace(op, [&] { op.setPaddingAttr(defaultAttr); });

      return success();
    }

    if (!op.getMask() || op.getOther())
      return failure();

    auto resultType = op.getResult().getType();
    auto elemType = getElementTypeOrSelf(resultType);

    // Set the default other field to 0.
    Value defaultOther;
    if (auto integerType = dyn_cast<IntegerType>(elemType)) {
      unsigned bitwidth = elemType.getIntOrFloatBitWidth();
      defaultOther = triton::createScalarOrSplatConstant(
          rewriter, op.getLoc(), resultType, APInt::getZero(bitwidth));
    }

    if (auto floatType = dyn_cast<FloatType>(elemType))
      defaultOther = triton::createScalarOrSplatConstant(
          rewriter, op.getLoc(), resultType,
          APFloat::getZero(floatType.getFloatSemantics()));

    assert(defaultOther && "fail to complement default other field.");
    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, op.getPtr(), op.getMask(), defaultOther, op.getCache(),
        op.getEvict(), op.getIsVolatile());
    return success();
  }
};

/// Insert `tensor.collapse_shape` to degenerate the dimensions of `tt.load` or
/// `tt.store` whose pointer shape contains one, for example: tensor<128x1xi32>
/// -> tensor<128xi32>.
template <typename OpTy>
class CanonicalizeTtPtrDimDegerationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType tensorType;
    auto loadOp = dyn_cast<triton::LoadOp>(*op);
    auto storeOp = dyn_cast<triton::StoreOp>(*op);
    if (loadOp) {
      tensorType =
          dyn_cast_if_present<RankedTensorType>(loadOp.getResult().getType());
    } else if (storeOp) {
      tensorType =
          dyn_cast_if_present<RankedTensorType>(storeOp.getValue().getType());
    } else {
      return failure();
    }
    if (!tensorType)
      return failure();

    Value ptr = op.getPtr();
    if (triton::isTensorPointerType(ptr.getType()))
      return failure();

    auto ptrType = dyn_cast_if_present<RankedTensorType>(ptr.getType());
    if (!ptrType)
      return failure();

    // Do NOT degenerate one size dimension when `%mask` exist because it will
    // cause lossing mask info and mask tracking failure.
    if (op.getMask())
      return failure();

    SmallVector<int64_t> oneSizeDims =
        getOneSizeDimIndices(tensorType.getShape());
    SmallVector<int64_t> degeneratedDims =
        getDegeneratableDimensions(op, oneSizeDims);
    if (degeneratedDims.empty())
      return failure();

    // Do NOT degenerate `tensor<1xT> or tensor<T>` into `T` as scalar math op
    // lowering is not yet supported perfectly.
    if ((tensorType.getRank() == oneSizeDims.size()) ||
        tensorType.getRank() == 0)
      return failure();

    SmallVector<int64_t> degeneratedShape(tensorType.getShape());
    for (auto dim : llvm::reverse(degeneratedDims)) {
      degeneratedShape.erase(degeneratedShape.begin() + dim);
    }

    Location loc = op.getLoc();
    if (degeneratedShape.empty()) {
      Value zeroValue =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      SmallVector<Value> zeroIdx(tensorType.getRank(), zeroValue);
      ptr = rewriter.create<tensor::ExtractOp>(loc, ptrType.getElementType(),
                                               ptr, zeroIdx);
      if (loadOp) {
        auto newLoadOp = rewriter.create<triton::LoadOp>(
            loc, ptr, loadOp.getCache(), loadOp.getEvict(),
            loadOp.getIsVolatile());
        auto splatOp =
            rewriter.create<triton::SplatOp>(loc, tensorType, newLoadOp);
        rewriter.replaceAllUsesWith(loadOp, splatOp);
        return success();
      } else if (storeOp) {
        auto value = rewriter.create<tensor::ExtractOp>(
            loc, tensorType.getElementType(), storeOp.getValue(), zeroIdx);
        rewriter.create<triton::StoreOp>(loc, ptr, value, storeOp.getCache(),
                                         storeOp.getEvict());
        rewriter.eraseOp(storeOp);
        return success();
      }
    }

    // Replace by new `tt.load` or `tt.store`.
    SmallVector<ReassociationIndices> reassocIndices =
        getReassociationsForFoldOneSizeDim(tensorType.getRank(),
                                           degeneratedDims);
    if (loadOp) {
      // Load degenerated shape then expand it to original shape of `tt.load`.
      SmallVector<Value, 3> newLoadOperands;
      for (auto opr : op->getOperands()) {
        auto oprType = cast<RankedTensorType>(opr.getType());
        auto newOprType =
            RankedTensorType::get(degeneratedShape, oprType.getElementType());
        newLoadOperands.emplace_back(rewriter.create<tensor::CollapseShapeOp>(
            loc, newOprType, opr, reassocIndices));
      }
      ptr = newLoadOperands[0];
      Value newLoadOp = rewriter.create<triton::LoadOp>(
          loc, ptr, loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());
      auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
          loc, tensorType, newLoadOp, reassocIndices);
      rewriter.replaceAllUsesWith(loadOp, expandShapeOp);
      return success();
    }
    if (storeOp) {
      // Collapse the shape of `%value` of `tt.store` then store it.
      SmallVector<Value, 3> newStoreOperands;
      for (auto opr : op->getOperands()) {
        auto oprType = cast<RankedTensorType>(opr.getType());
        auto newOprType =
            RankedTensorType::get(degeneratedShape, oprType.getElementType());
        newStoreOperands.emplace_back(rewriter.create<tensor::CollapseShapeOp>(
            loc, newOprType, opr, reassocIndices));
      }
      ptr = newStoreOperands[0];
      Value value = newStoreOperands[1];
      rewriter.create<triton::StoreOp>(loc, ptr, value, op.getCache(),
                                       op.getEvict());
      rewriter.eraseOp(storeOp);
      return success();
    }

    // Default return if neither `tt.load` nor `tt.store`.
    return failure();
  }
};

/// Replace tensor pointer's shape of `tt.load` or `tt.store` to degenerate the
/// one size dimension. For example:
/// ``` mlir
///  %ptr = tt.make_tensor_ptr [...], [...], [...] : !tt.ptr<tensor<128x1xi32>>
///  %res = tt.load %ptr : !tt.ptr<tensor<128x1xi32>>
/// ```
/// is converted to:
/// ``` mlir
///  %ptr = tt.make_tensor_ptr [...], [...], [...] : !tt.ptr<tensor<128xi32>>
///  %res = tt.load %ptr : !tt.ptr<tensor<128xi32>>
///  %eps = tensor.expand_shape [...] : tensor<128xi32> into tensor<128x1xi32>
/// ```
template <typename OpTy>
class CanonicalizeTtTensorPtrDimDegerationPattern
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loadOp = dyn_cast<triton::LoadOp>(*op);
    auto storeOp = dyn_cast<triton::StoreOp>(*op);
    if (!loadOp && !storeOp) {
      return failure();
    }

    Value ptr = op.getPtr();
    if (!triton::isTensorPointerType(ptr.getType()))
      return failure();
    Type pointeeType =
        dyn_cast<triton::PointerType>(ptr.getType()).getPointeeType();
    RankedTensorType tensorType =
        dyn_cast_if_present<RankedTensorType>(pointeeType);
    if (!tensorType)
      return failure();

    // No need to handle `tt.advance` because `ptr-strength-reduction` pass will
    // eliminate it.
    auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>();
    if (!makeTensorPtrOp || op.getMask())
      return failure();

    SmallVector<int64_t> oneSizeDims =
        getOneSizeDimIndices(tensorType.getShape());
    SmallVector<int64_t> degeneratedDims =
        getDegeneratableDimensions(op, oneSizeDims);

    // Do NOT degenerate `tensor<1xT> or tensor<T>` into `T` as scalar math op
    // lowering is not yet supported perfectly.
    if ((tensorType.getRank() == oneSizeDims.size()) ||
        tensorType.getRank() == 0)
      return failure();

    SmallVector<Value> strides(makeTensorPtrOp.getStrides());
    SmallVector<Value> offsets(makeTensorPtrOp.getOffsets());

    // The `offset` is required for calculating the base address offset, so it
    // cannot be removed. The division of this dimension has no effect on the
    // base address offset, meaning the `stride` or `offset` for this dimension
    // is 0.
    const auto checkNotDegeneratedDim = [&offsets, &strides](const int idx) {
      auto offset = getConstantIntValue(offsets[idx]);
      auto stride = getConstantIntValue(strides[idx]);
      return (!offset.has_value() || *offset != 0) &&
             (!stride.has_value() || *stride != 0);
    };
    degeneratedDims.erase(std::remove_if(degeneratedDims.begin(),
                                         degeneratedDims.end(),
                                         checkNotDegeneratedDim),
                          degeneratedDims.end());
    if (degeneratedDims.empty())
      return failure();

    SmallVector<int64_t> degeneratedShape(tensorType.getShape());
    for (auto dim : llvm::reverse(degeneratedDims)) {
      degeneratedShape.erase(degeneratedShape.begin() + dim);
    }

    SmallVector<int32_t> degeneratedBoundaryCheck =
        getDegeneratedBoundaryCheck(op, degeneratedDims);

    // Load one element block ptr then splat it to a tensor whose shape looks
    // like `tensor<1x1x...1xT>`, otherwise create a new degenerated ptr.
    Location loc = op.getLoc();
    Value base = makeTensorPtrOp.getBase();
    if (degeneratedShape.empty()) {
      Value zeroValue =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      SmallVector<Value> zeroIdx(tensorType.getRank(), zeroValue);
      if (loadOp) {
        Value newLoadOp = rewriter.create<triton::LoadOp>(
            loc, base, degeneratedBoundaryCheck, loadOp.getPadding(),
            loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
        Value splatOp =
            rewriter.create<triton::SplatOp>(loc, tensorType, newLoadOp);
        rewriter.replaceAllUsesWith(loadOp, splatOp);
        return success();
      } else if (storeOp) {
        Value value = storeOp.getValue();
        value = rewriter.create<tensor::ExtractOp>(
            loc, tensorType.getElementType(), value, zeroIdx);
        rewriter.create<triton::StoreOp>(
            loc, base, value, degeneratedBoundaryCheck, storeOp.getCache(),
            storeOp.getEvict());
        rewriter.eraseOp(storeOp);
        return success();
      }
    }

    // Create new dimension degenerated `tt.make_tensor_ptr`. Here we only need
    // to handle the `tt.make_tensor_ptr` pointer because
    // `ptr-strength-reduction` pass will eliminate `tt.advance`.
    SmallVector<Value> shape(makeTensorPtrOp.getShape());
    SmallVector<int32_t> order(makeTensorPtrOp.getOrder());
    for (auto dim : llvm::reverse(degeneratedDims)) {
      shape.erase(shape.begin() + dim);
      strides.erase(strides.begin() + dim);
      offsets.erase(offsets.begin() + dim);
      order.erase(llvm::find(order, dim));
      for (uint64_t idx = 0; idx < order.size(); ++idx) {
        if (order[idx] > dim)
          order[idx] = order[idx] - 1;
      }
    }
    auto newTensorType =
        RankedTensorType::get(degeneratedShape, tensorType.getElementType());
    uint32_t ptrAddrSpace =
        cast<triton::PointerType>(ptr.getType()).getAddressSpace();
    Type makeTensorPtrType =
        triton::PointerType::get(newTensorType, ptrAddrSpace);
    ptr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, makeTensorPtrType, base, shape, strides, offsets, order);

    // Replace by new `tt.load` or `tt.store`.
    SmallVector<ReassociationIndices> reassocIndices =
        getReassociationsForFoldOneSizeDim(tensorType.getRank(),
                                           degeneratedDims);
    if (loadOp) {
      // Load degenerated shape then expand it to original shape of `tt.load`.
      Value mask = loadOp.getMask();
      if (mask) {
        auto maskType = dyn_cast_if_present<RankedTensorType>(mask.getType());
        auto collapsedMaskType =
            RankedTensorType::get(degeneratedShape, maskType.getElementType());
        mask = rewriter.create<tensor::CollapseShapeOp>(loc, collapsedMaskType,
                                                        mask, reassocIndices);
      }
      Value other = loadOp.getOther();
      if (other) {
        other = rewriter.create<tensor::CollapseShapeOp>(loc, newTensorType,
                                                         other, reassocIndices);
      }
      Value newLoadOp = rewriter.create<triton::LoadOp>(
          loc, ptr, mask, other, degeneratedBoundaryCheck, loadOp.getPadding(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
          loc, tensorType, newLoadOp, reassocIndices);
      rewriter.replaceAllUsesWith(loadOp, expandShapeOp);
      return success();
    }
    if (storeOp) {
      // Collapse the shape of `%value` of `tt.store` then store it.
      Value value = storeOp.getValue();
      auto collapsedValueType =
          RankedTensorType::get(degeneratedShape, tensorType.getElementType());
      value = rewriter.create<tensor::CollapseShapeOp>(loc, collapsedValueType,
                                                       value, reassocIndices);
      rewriter.create<triton::StoreOp>(loc, ptr, value,
                                       degeneratedBoundaryCheck, op.getCache(),
                                       op.getEvict());
      rewriter.eraseOp(storeOp);
      return success();
    }
    // Default return if neither `tt.load` nor `tt.store`.
    return failure();
  }
};

/// Create a smaller shape `tt.make_tensor_ptr` to degenerate the constancy of
/// the block pointer then broadcast the smaller result of `tt.load` to
/// original shape. For example:
/// ``` mlir
/// %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64, %c64_i64],
///                                [%c0_i64, %c8_i64, %c0_i64],
///                                [%c0_i32, %c0_i32, %c0_i32]
///      {order = array<i32: 2, 0, 1>} : !tt.ptr<tensor<64x64x64xf32>>
/// %1 = tt.load %0 : !tt.ptr<tensor<64x64x64xf32>>
/// ```
/// is converted to:
/// ``` mlir
/// %0 = tt.make_tensor_ptr %arg0, [%c64_i64],
///                                [%c8_i64],
///                                [%c0_i32]
///      {order = array<i32: 0>} : !tt.ptr<tensor<64xf32>>
/// %1 = tt.load %0 : !tt.ptr<tensor<64xf32>>
/// %expanded = tensor.expand_shape %1 [[0, 1, 2]] : tensor<64xf32> into
///             tensor<1x64x1xf32>
/// %2 = tt.broadcast %expanded : tensor<1x64x1xf32> -> tensor<64x64x64xf32>
/// ```
///
/// If boundary check is necessary, set the stride's value to one after
/// constancy coalesced. This pattern is pre-requisite for `triton-to-linalg`
/// conversion as the conversion will treat one size dimension without boundary
/// check as broadcastable. For Example:
/// ``` mlir
///  %ptr = tt.make_tensor_ptr [...], [0, 0], [...] : !tt.ptr<tensor<16x16xi32>>
///  %res = tt.load %ptr {boundaryCheck = <1>} : !tt.ptr<tensor<16x16xi32>>
/// ```
/// is converted to:
/// ``` mlir
///  %ptr = tt.make_tensor_ptr [...], [0, 1], [...] : !tt.ptr<tensor<1x1xi32>>
///  %res = tt.load %ptr {boundaryCheck = <1>} : !tt.ptr<tensor<1x1xi32>>
///  %bct = tt.broadcast %res : tensor<1x1xi32> -> tensor<16x16xi32>
/// ```
template <typename OpTy>
class CanonicalizeTtTensorPtrConstancyDegerationPattern
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loadOp = dyn_cast<triton::LoadOp>(*op);
    auto storeOp = dyn_cast<triton::StoreOp>(*op);
    if (!loadOp && !storeOp) {
      return failure();
    }

    Value ptr = op.getPtr();
    if (!triton::isTensorPointerType(ptr.getType()))
      return failure();
    Type pointeeType =
        dyn_cast<triton::PointerType>(ptr.getType()).getPointeeType();
    RankedTensorType tensorType =
        dyn_cast_if_present<RankedTensorType>(pointeeType);
    if (!tensorType)
      return failure();

    // No need to handle `tt.advance` because `ptr-strength-reduction` pass will
    // eliminate it.
    auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>();
    if (!makeTensorPtrOp || op.getMask())
      return failure();

    // Check if `%ptr` is coalescable and calculate the coalesced shape.
    ArrayRef<int64_t> tensorShape = tensorType.getShape();
    SmallVector<int64_t> coalescedShape(tensorShape);
    SmallVector<int64_t> coalescedDims;
    SmallVector<Value> blockPtrStrides = makeTensorPtrOp.getStrides();
    for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
      llvm::APInt strideVal;
      if (!matchPattern(blockPtrStrides[dim], m_ConstantInt(&strideVal)))
        continue;
      if (strideVal.getSExtValue() == 0 && tensorShape[dim] != 1) {
        coalescedDims.push_back(dim);
        coalescedShape[dim] = 1;
      }
    }
    if (coalescedDims.empty())
      return failure();

    // Set boundary checked dimension's stride value to one.
    Location loc = op.getLoc();
    std::optional<ArrayRef<int32_t>> boundaryCheck = op.getBoundaryCheck();
    if (boundaryCheck.has_value() && !boundaryCheck.value().empty()) {
      for (int64_t dim : coalescedDims) {
        if (llvm::any_of(boundaryCheck.value(),
                         [&dim](int32_t bc) { return dim == bc; })) {
          blockPtrStrides[dim] = rewriter.create<arith::ConstantIntOp>(
              loc, 1, rewriter.getI64Type());
        }
      }
    }

    // Create new `tt.make_tensor_ptr` to coalesce the constancy shape.
    uint32_t ptrAddrSpace =
        cast<triton::PointerType>(ptr.getType()).getAddressSpace();
    auto newPtrType =
        RankedTensorType::get(coalescedShape, tensorType.getElementType());
    Type newMakeTensorPtrType =
        triton::PointerType::get(newPtrType, ptrAddrSpace);
    ptr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, newMakeTensorPtrType, makeTensorPtrOp.getBase(),
        makeTensorPtrOp.getShape(), blockPtrStrides,
        makeTensorPtrOp.getOffsets(), makeTensorPtrOp.getOrder());

    // Create new coalesced shape `tt.load` or `tt.store`.
    if (loadOp) {
      // Load coalesced shape then expand it to original shape of `tt.load`.
      auto newLoadOp = rewriter.create<triton::LoadOp>(
          loc, ptr, boundaryCheck.value(), loadOp.getPadding(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      auto broadcastOp =
          rewriter.create<triton::BroadcastOp>(loc, tensorType, newLoadOp);
      rewriter.replaceAllUsesWith(loadOp, broadcastOp);
      return success();
    }
    if (storeOp) {
      // Coalesce the shape of `%value` of `tt.store` then store it.
      Value value = storeOp.getValue();
      IntegerAttr zeroAttr = rewriter.getIndexAttr(0);
      IntegerAttr oneAttr = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> offsets(tensorType.getRank(), zeroAttr);
      SmallVector<OpFoldResult> sizes(tensorType.getRank(), oneAttr);
      SmallVector<OpFoldResult> strides(tensorType.getRank(), oneAttr);
      for (auto [idx, size] : llvm::enumerate(sizes)) {
        sizes[idx] = rewriter.getIndexAttr(coalescedShape[idx]);
      }
      for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
        strides[dim] = rewriter.getIndexAttr(tensorType.getShape()[dim] /
                                             coalescedShape[dim]);
      }
      value = rewriter.create<tensor::ExtractSliceOp>(loc, value, offsets,
                                                      sizes, strides);
      rewriter.create<triton::StoreOp>(loc, ptr, value, boundaryCheck.value(),
                                       op.getCache(), op.getEvict());
      rewriter.eraseOp(storeOp);
      return success();
    }

    // Default return if neither `tt.load` nor `tt.store`.
    return failure();
  }
};

/// This pattern applies to memory access operations like `tt.load` and
/// `tt.store`. If their mask is created through a scalar splat of `i1` or used
/// as an input for `arith.andi` or `arith.ori`, it results in subsequent
/// conversion to discrete memory access operations. This pattern converts
/// memory access operations with this specific mask into `scf.if` + the memory
/// access operation(without splat i1 mask).
///
/// Case1: If the mask is created by `arith.andi`, it convert to:
/// ```
///   if (%cond) {
///       yield tt.load
///   } else {
///       yield other operand from tt.load op
///   }
/// ```
///
/// Case2: If the mask is created by `arith.ori`, it convert to:
/// ```
///   if (%cond) {
///       yield other operand from tt.load op
///   } else {
///       yield tt.load
///   }
/// ```
///
/// Example 1:
///
/// ``` mlir
///   %mask0 = tt.splat %bool : i1 -> tensor<1x1024xi1>
///   %mask1 = ...
///   %mask = arith.andi %mask0, %mask1
///   %res = tt.load %ptr, %mask : tensor<1x1024x!tt.ptr<f32>>
///   tt.return %res : tensor<1x1024xf32>
/// ```
/// is converted to:
/// ``` mlir
///   %constant = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
///   %res = scf.if (%bool) -> tensor<1x1024xf32> {
///     %load = tt.load %ptr, %mask1 : tensor<1x1024x!tt.ptr<f32>>
///     scf.yield %load : tensor<1x1024xf32>
///   } else {
///     scf.yield %constant : tensor<1x1024xf32>
///   }
///   tt.return %res : tensor<1x1024xf32>
/// ```
///
/// Example 2:
///
/// ``` mlir
///   %other = arith.constant dense<1.000000e+00> : tensor<1x1024xf32>
///   %mask0 = tt.splat %bool : i1 -> tensor<1x1024xi1>
///   %mask1 = ...
///   %mask = arith.andi %mask0, %mask1
///   %res = tt.load %ptr, %mask, %other : tensor<1x1024x!tt.ptr<f32>>
///   tt.return %res : tensor<1x1024xf32>
/// ```
/// is converted to:
/// ``` mlir
///   %other = arith.constant dense<1.000000e+00> : tensor<1x1024xf32>
///   %res = scf.if (%bool) -> tensor<1x1024xf32> {
///     %load = tt.load %ptr, %mask1, %other : tensor<1x1024x!tt.ptr<f32>>
///     scf.yield %load : tensor<1x1024xf32>
///   } else {
///     scf.yield %other : tensor<1x1024xf32>
///   }
///   tt.return %res : tensor<1x1024xf32>
/// ```
///
/// Example 3:
///
/// ``` mlir
///   %mask = tt.splat %bool : i1 -> tensor<1x1024xi1>
///   tt.store %ptr, %val, %mask : tensor<1x1024x!tt.ptr<f32>>
/// ```
/// is converted to:
/// ``` mlir
///   scf.if (%bool) {
///     tt.store %ptr, %val : tensor<1x1024x!tt.ptr<f32>>
///   }
/// ```
///
/// Example 4:
///
/// ``` mlir
///   %mask = tt.splat %bool : i1 -> tensor<64x64xi1>
///   %res = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val, %mask :
///       (tensor<64x64x!tt.ptr<f32>>, tensor<64x64xf32>, tensor<64x64xi1>)
///        -> tensor<64x64xf32>
///   tt.return %res : tensor<64x64xf32>
/// ```
/// is converted to:
/// ``` mlir
///   %constant = arith.constant dense<0.000000e+00> : tensor<64x64xi1>
///   %res = scf.if (%bool) -> tensor<64x64xf32> {
///     %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val :
///         (tensor<64x64x!tt.ptr<f32>>, tensor<64x64xf32>) -> tensor<64x64xf32>
///     scf.yield %atomic : tensor<64x64xf32>
///   } else {
///     scf.yield %constant : tensor<64x64xf32>
///   }
///   tt.return %res : tensor<64x64xf32>
/// ```
template <typename OpTy>
class CanonicalizeTtMaskAccessPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // This pattern only support load, store and atomic_rmw op.
    if (!(llvm::isa<triton::LoadOp>(op) || llvm::isa<triton::StoreOp>(op) ||
          llvm::isa<triton::AtomicRMWOp>(op)))
      return failure();

    MaskInfo maskInfo = getMaskInfo(op.getMask(), rewriter);

    if (!maskInfo.cond)
      return failure();

    assert(maskInfo.logicalType != MaskInfo::LogicalType::NONE);
    auto loc = op->getLoc();
    auto resTypes = op->getResultTypes();
    scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc,
                                                /*resultTypes*/ resTypes,
                                                /*cond*/ maskInfo.cond,
                                                /*addElseBlock*/ true);

    // Create new op in then region if logical type is AND.
    // Then region: new op(new_mask) + yield(if new op has result)
    if (maskInfo.logicalType == MaskInfo::LogicalType::AND) {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    } else {
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    }
    if (maskInfo.mask) {
      op.getMaskMutable().assign(maskInfo.mask);
    } else {
      op.getMaskMutable().clear();
    }
    auto newOp = rewriter.clone(*op);
    if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(*newOp)) {
      // If load op has no mask, delete other operand.
      if (op.getMaskMutable().empty())
        loadOp.getOtherMutable().clear();
    }

    if (resTypes.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.create<scf::YieldOp>(loc, newOp->getResults());
      // If else region is empty, it will be fold in canonicalize.
      // Else region: constant(0 or other value) + yield if logical type is AND.
      if (maskInfo.logicalType == MaskInfo::LogicalType::AND) {
        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      } else {
        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      }
      // If the processed op has results, it will has only one result.
      Value yieldVal = nullptr;
      // If load has other arg set, use other value
      auto loadOp = llvm::dyn_cast<triton::LoadOp>(op.getOperation());
      if (loadOp && loadOp.getOther()) {
        yieldVal = loadOp.getOther();
      } else {
        yieldVal = rewriter.create<arith::ConstantOp>(
            loc, resTypes.front(), rewriter.getZeroAttr(resTypes.front()));
      }

      rewriter.create<scf::YieldOp>(loc, yieldVal);
      rewriter.replaceOp(op, ifOp);
    }

    return success();
  }
};

class CanonicalizeTtAssertPattern : public OpRewritePattern<triton::AssertOp> {
public:
  explicit CanonicalizeTtAssertPattern(MLIRContext *ctx)
      : OpRewritePattern<triton::AssertOp>(ctx) {}

  LogicalResult matchAndRewrite(triton::AssertOp op,
                                PatternRewriter &rewriter) const override {
    auto condVal = op.getCondition();
    auto valType = condVal.getType();
    auto rank = valType.getRank();
    auto assertMessage =
        llvm::formatv("{0}:{1}: {2} Assertion `{3}` failed", op.getFile(),
                      op.getLine(), op.getFunc(), op.getMessage());
    assert(isa<mlir::IntegerType>(valType.getElementType()) &&
           "Only support int tensor for assert");
    // If the AssertOp input shape dimension is 1 or 0 dimension and the 0th
    // dimension is 1, it is converted to ScalarAssertOp.
    if ((rank != 1 && rank != 0) || (rank > 0 && valType.getShape()[0] != 1)) {
      return failure();
    }
    auto rankType = cast<RankedTensorType>(valType);
    auto elemType = rankType.getElementType();
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value cond;
    if (rank == 0) {
      cond =
          rewriter.create<tensor::ExtractOp>(op.getLoc(), condVal).getResult();
    } else {
      cond = rewriter.create<tensor::ExtractOp>(op.getLoc(), condVal, zeroIndex)
                 .getResult();
    }
    // If the input data type is not i1, cast it to i1.
    if (!elemType.isInteger(1)) {
      Value zero = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIntegerAttr(elemType, 0));
      cond = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, cond, zero);
    }
    rewriter.create<triton::linalg_ext::ScalarAssertOp>(op.getLoc(), cond,
                                                        assertMessage.str());
    rewriter.eraseOp(op);
    return success();
  }
};

struct CanonicalizeTritonPass
    : public CanonicalizeTritonBase<CanonicalizeTritonPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::linalg_ext::LinalgExtDialect, tensor::TensorDialect,
                    scf::SCFDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    // Canonicalize mask-related ops and its connection.
    getOperation().walk([&](Operation *op) {
      // Only support arth.andi connections now.
      if (auto andOp = llvm::dyn_cast<arith::AndIOp>(op)) {
        MaskOpsOrganizer organizer(andOp);
        organizer.reorganize();
      }
      return WalkResult::advance();
    });

    RewritePatternSet patterns(&ctx);
    patterns.insert<
        CanonicalizeTtBroadCastPattern, CanonicalizeTtLoadPattern,
        CanonicalizeTtMaskAccessPattern<triton::LoadOp>,
        CanonicalizeTtMaskAccessPattern<triton::StoreOp>,
        CanonicalizeTtMaskAccessPattern<triton::AtomicRMWOp>,
        CanonicalizeTtAssertPattern,
        CanonicalizeTtPtrDimDegerationPattern<triton::LoadOp>,
        CanonicalizeTtPtrDimDegerationPattern<triton::StoreOp>,
        CanonicalizeTtTensorPtrConstancyDegerationPattern<triton::LoadOp>,
        CanonicalizeTtTensorPtrConstancyDegerationPattern<triton::StoreOp>,
        CanonicalizeTtTensorPtrDimDegerationPattern<triton::LoadOp>,
        CanonicalizeTtTensorPtrDimDegerationPattern<triton::StoreOp>>(&ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
std::unique_ptr<Pass> createCanonicalizeTritonPass() {
  return std::make_unique<CanonicalizeTritonPass>();
}

} // namespace triton
} // namespace mlir
