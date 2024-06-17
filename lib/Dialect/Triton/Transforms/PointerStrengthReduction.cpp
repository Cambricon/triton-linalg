//===- PointerStrengthReduction.cpp - reduce pointer strength ---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <memory>
#include <optional>
#include <stddef.h>
#include <stdint.h>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-linalg/Dialect/Triton/Transforms/PassDetail.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ptr-strength-reduction"

using namespace mlir;
using namespace triton;

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace {

/// Check if there are repeat arguments in block inputs and terminatorOp.
static bool verifyArgsMatchTerminatorInputsInBlock(Block *block,
                                                   unsigned blockArgStart,
                                                   unsigned terminalArgStart,
                                                   unsigned len) {
  auto *terminatorOp = block->getTerminator();
  if (!terminatorOp) {
    return false;
  }
  for (unsigned i = 0; i < len; i++) {
    if (block->getArgument(blockArgStart + i) !=
        terminatorOp->getOperand(terminalArgStart + i)) {
      return false;
    }
  }
  return true;
}

/// Reconstruct ptr computation in block.
static SmallVector<Value> reconstructPtrInBlock(Block *block, unsigned argIdx,
                                                unsigned startIdx,
                                                const PtrInfo &info,
                                                PatternRewriter &rewriter,
                                                bool hasPtr = true) {
  SmallVector<Value> ret;
  Value ptr;
  if (hasPtr) {
    ptr = block->insertArgument(startIdx++, info.ptr().getType(),
                                info.ptr().getLoc());
    ret.push_back(ptr);
  } else {
    ptr = info.ptr();
  }
  auto loc = block->begin()->getLoc();
  rewriter.setInsertionPointToStart(block);
  auto rawArgVal = block->getArgument(argIdx);
  if (!info.isBlockPtr()) {
    auto blockArgOffset = block->insertArgument(
        startIdx++, info.offset().getType(), info.offset().getLoc());
    ret.push_back(blockArgOffset);
    // Reconstruct splat and addptr.
    Value newPtr =
        rewriter.create<triton::SplatOp>(loc, rawArgVal.getType(), ptr)
            .getResult();
    newPtr = rewriter
                 .create<triton::AddPtrOp>(loc, newPtr.getType(), newPtr,
                                           blockArgOffset)
                 .getResult();
    rewriter.replaceAllUsesWith(rawArgVal, newPtr);
  } else {
    SmallVector<Value> sizes;
    for (auto size : info.sizes()) {
      auto val =
          block->insertArgument(startIdx++, size.getType(), size.getLoc());
      ret.push_back(val);
      sizes.push_back(val);
    }
    SmallVector<Value> strides;
    for (auto stride : info.strides()) {
      auto val =
          block->insertArgument(startIdx++, stride.getType(), stride.getLoc());
      ret.push_back(val);
      strides.push_back(val);
    }
    SmallVector<Value> offsets;
    for (auto offset : info.offsets()) {
      auto val =
          block->insertArgument(startIdx++, offset.getType(), offset.getLoc());
      ret.push_back(val);
      offsets.push_back(val);
    }
    auto newPtr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, rawArgVal.getType(), info.ptr(), sizes, strides, offsets,
        info.order());
    rewriter.replaceAllUsesWith(rawArgVal, newPtr);
  }
  return ret;
}

/// Reconstruct ptr computation after target op.
static void reconstructAfterOp(Operation *op, Value rawPtr, unsigned startIdx,
                               bool hasPtr, const PtrInfo &info,
                               PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(op);
  if (!info.isBlockPtr()) {
    Value ptr, offset;
    if (hasPtr) {
      ptr = op->getResult(startIdx);
      offset = op->getResult(startIdx + 1);
    } else {
      ptr = info.ptr();
      offset = op->getResult(startIdx);
    }
    Value newptr =
        rewriter.create<triton::SplatOp>(op->getLoc(), rawPtr.getType(), ptr)
            .getResult();
    newptr = rewriter
                 .create<triton::AddPtrOp>(op->getLoc(), rawPtr.getType(),
                                           newptr, offset)
                 .getResult();
    rewriter.replaceAllUsesWith(rawPtr, newptr);
  } else {
    Value ptr;
    if (hasPtr) {
      ptr = op->getResult(startIdx++);
    } else {
      ptr = info.ptr();
    }

    SmallVector<Value> sizes;
    for (size_t i = 0; i < info.sizes().size(); i++) {
      sizes.push_back(op->getResult(startIdx++));
    }
    SmallVector<Value> strides;
    for (size_t i = 0; i < info.strides().size(); i++) {
      strides.push_back(op->getResult(startIdx++));
    }
    SmallVector<Value> offsets;
    for (size_t i = 0; i < info.offsets().size(); i++) {
      offsets.push_back(op->getResult(startIdx++));
    }
    auto newPtr = rewriter.create<triton::MakeTensorPtrOp>(
        op->getLoc(), rawPtr.getType(), ptr, sizes, strides, offsets,
        info.order());
    rewriter.replaceAllUsesWith(rawPtr, newPtr);
  }
}

/// Pack and return all ptr infos.
static SmallVector<Value> packPtrInfo(const PtrInfo &info,
                                      bool ignorePtr = false) {
  SmallVector<Value> ret;
  if (!ignorePtr) {
    ret.push_back(info.ptr());
  }
  if (!info.isBlockPtr()) {
    ret.push_back(info.offset());
  } else {
    for (auto size : info.sizes()) {
      ret.push_back(size);
    }
    for (auto stride : info.strides()) {
      ret.push_back(stride);
    }
    for (auto offset : info.offsets()) {
      ret.push_back(offset);
    }
  }
  return ret;
}

/// Structure info for region successors.
struct SuccessorInfo {
  SmallVector<RegionSuccessor, 2> successors;
  Region *region;
  bool isLoopLike = false;

  unsigned getResultsSize() const {
    auto *terminatorOp = region->front().getTerminator();
    return terminatorOp->getNumOperands();
  }

  // Note:
  // - we only return the size of the first block arguments.
  unsigned getArgumentSize() const { return region->front().getNumArguments(); }
};

/// Reducing strength of computation with pointers in Triton IR.
/// The algorithm proceeds as follows:
/// - Step 1: Identify the target operation and match the target pointer
///   (e.g., those produced by AddPtrOp or SplatOp).
/// - Step 2: Execute the appropriate regularization operations.
/// - Step 3: Apply these patterns greedily.
///
/// For example, let's consider the operation 'triton::AddPtrOp':
///
/// ```
/// %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
/// %3 = tt.addptr %2, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
/// ```
///
/// After conversion, the IR is generated as follows:
///
/// ```
/// %2 = arith.addi %0, %cst : tensor<128xi32>
/// %3 = tt.addptr %1, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
/// ```
///
/// All offsets will be accumulated.
template <typename OpT>
class PtrStrengthReductionPattern : public OpRewritePattern<OpT> {
public:
  explicit PtrStrengthReductionPattern(MLIRContext *ctx)
      : OpRewritePattern<OpT>(ctx) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
                   << "[PtrStrengthReductionPattern not support!] Op:" << op;);
    return failure();
  }

private:
  FailureOr<PtrInfo> getAddPtrAndOffset(Value ptr) const {
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    if (!addPtrOp) {
      return failure();
    }
    return PtrInfo(addPtrOp.getPtr(), addPtrOp.getOffset());
  }

  FailureOr<PtrInfo> getSplatPtrAndOffset(Value ptr) const {
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    if (addPtrOp) {
      auto splatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
      if (splatOp) {
        return PtrInfo(splatOp.getSrc(), addPtrOp.getOffset());
      }
    }
    return failure();
  }

  FailureOr<PtrInfo> getPreviousBlockPtrAndOffset(Value ptr) const {
    auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>();
    if (!makeTensorPtrOp) {
      return failure();
    }
    IRRewriter rewriter(ptr.getContext());
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    SmallVector<Value> offsets;
    SmallVector<int32_t> orders;
    for (int i = 0; i < makeTensorPtrOp.getOffsets().size(); ++i) {
      sizes.push_back(makeTensorPtrOp.getShape()[i]);
      strides.push_back(makeTensorPtrOp.getStrides()[i]);
      offsets.push_back(makeTensorPtrOp.getOffsets()[i]);
      orders.push_back(makeTensorPtrOp.getOrder()[i]);
    }
    return PtrInfo(makeTensorPtrOp.getBase(), sizes, strides, offsets, orders);
  }

  template <typename OperationT>
  LogicalResult normalizeWithSplat(Value ptr, OperationT op,
                                   PatternRewriter &rewriter) const {
    auto splatOp = ptr.getDefiningOp<triton::SplatOp>();
    if (!splatOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getResult().getType(),
                                                 splatOp.getSrc());
    return success();
  }

  Value accOffset(Location loc, Value infoOffset, Value currentOffset,
                  PatternRewriter &rewriter) const {
    Type offsetElementType = getElementTypeOrSelf(infoOffset.getType());
    Type currentOffsetElementType =
        getElementTypeOrSelf(currentOffset.getType());
    // Promote offset type to the type of largest bitwith.
    if (offsetElementType.getIntOrFloatBitWidth() >
        currentOffsetElementType.getIntOrFloatBitWidth()) {
      currentOffset = rewriter.createOrFold<arith::ExtSIOp>(
          loc, infoOffset.getType(), currentOffset);
    } else if (offsetElementType.getIntOrFloatBitWidth() <
               currentOffsetElementType.getIntOrFloatBitWidth()) {
      infoOffset = rewriter.createOrFold<arith::ExtSIOp>(
          loc, currentOffset.getType(), infoOffset);
    }

    auto newOffset =
        rewriter.createOrFold<arith::AddIOp>(loc, currentOffset, infoOffset);
    return newOffset;
  }
};

template <>
LogicalResult PtrStrengthReductionPattern<triton::AdvanceOp>::matchAndRewrite(
    triton::AdvanceOp op, PatternRewriter &rewriter) const {
  auto info = getPreviousBlockPtrAndOffset(op.getPtr());
  if (failed(info)) {
    return failure();
  }
  auto loc = op.getLoc();
  SmallVector<Value> offsets;
  for (unsigned i = 0; i < info->offsets().size(); i++) {
    offsets.push_back(
        accOffset(loc, info->offset(i), op.getOffsets()[i], rewriter));
  }
  rewriter.replaceOpWithNewOp<triton::MakeTensorPtrOp>(
      op, op.getResult().getType(), info->ptr(), info->sizes(), info->strides(),
      offsets, info->order());
  return success();
}

template <>
LogicalResult PtrStrengthReductionPattern<triton::AddPtrOp>::matchAndRewrite(
    triton::AddPtrOp op, PatternRewriter &rewriter) const {
  auto info = getAddPtrAndOffset(op.getPtr());
  if (failed(info)) {
    return failure();
  }
  auto loc = op.getLoc();
  Value currentOffset = op.getOffset();
  auto newOffset = accOffset(loc, info->offset(), currentOffset, rewriter);
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, info->ptr().getType(),
                                                info->ptr(), newOffset);
  return success();
}

template <>
LogicalResult
PtrStrengthReductionPattern<triton::ExpandDimsOp>::matchAndRewrite(
    triton::ExpandDimsOp op, PatternRewriter &rewriter) const {
  auto info = getAddPtrAndOffset(op.getOperand());
  if (failed(info)) {
    return normalizeWithSplat(op.getOperand(), op, rewriter);
  }
  auto loc = op.getLoc();
  auto newOffset =
      rewriter.create<triton::ExpandDimsOp>(loc, info->offset(), op.getAxis());
  auto newPtr =
      rewriter.create<triton::ExpandDimsOp>(loc, info->ptr(), op.getAxis());
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newPtr.getType(), newPtr,
                                                newOffset);
  return success();
}

template <>
LogicalResult PtrStrengthReductionPattern<triton::BroadcastOp>::matchAndRewrite(
    triton::BroadcastOp op, PatternRewriter &rewriter) const {
  auto info = getAddPtrAndOffset(op.getOperand());
  if (failed(info)) {
    return normalizeWithSplat(op.getOperand(), op, rewriter);
  }
  auto loc = op.getLoc();
  auto newOffset = rewriter.create<triton::BroadcastOp>(
      loc,
      RankedTensorType::get(
          op.getResult().getType().cast<ShapedType>().getShape(),
          getElementTypeOrSelf(info->offset().getType())),
      info->offset());
  auto newPtr = rewriter.create<triton::BroadcastOp>(
      loc, op.getResult().getType(), info->ptr());
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newPtr.getType(), newPtr,
                                                newOffset);
  return success();
}

template <>
LogicalResult PtrStrengthReductionPattern<triton::ReshapeOp>::matchAndRewrite(
    triton::ReshapeOp op, PatternRewriter &rewriter) const {
  auto info = getAddPtrAndOffset(op.getOperand());
  if (failed(info)) {
    return normalizeWithSplat(op.getOperand(), op, rewriter);
  }
  auto loc = op.getLoc();
  auto newOffset = rewriter.create<triton::ReshapeOp>(
      loc,
      RankedTensorType::get(
          op.getResult().getType().cast<ShapedType>().getShape(),
          getElementTypeOrSelf(info->offset().getType())),
      info->offset(), false);
  auto newPtr = rewriter.create<triton::ReshapeOp>(
      loc, op.getResult().getType(), info->ptr(), false);
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newPtr.getType(), newPtr,
                                                newOffset);
  return success();
}

template <>
LogicalResult PtrStrengthReductionPattern<triton::TransOp>::matchAndRewrite(
    triton::TransOp op, PatternRewriter &rewriter) const {
  auto info = getAddPtrAndOffset(op.getOperand());
  if (failed(info)) {
    return normalizeWithSplat(op.getOperand(), op, rewriter);
  }
  auto loc = op.getLoc();
  auto newOffset = rewriter.create<triton::TransOp>(
      loc,
      RankedTensorType::get(
          op.getResult().getType().cast<ShapedType>().getShape(),
          getElementTypeOrSelf(info->offset().getType())),
      info->offset(), op.getOrder());
  auto newPtr = rewriter.create<triton::TransOp>(loc, op.getResult().getType(),
                                                 info->ptr(), op.getOrder());
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newPtr.getType(), newPtr,
                                                newOffset);
  return success();
}

template <>
LogicalResult PtrStrengthReductionPattern<triton::BitcastOp>::matchAndRewrite(
    triton::BitcastOp op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto info = getSplatPtrAndOffset(op.getSrc());
  if (failed(info)) {
    auto splatOp = op.getSrc().getDefiningOp<triton::SplatOp>();
    if (splatOp) {
      auto rawPtr = rewriter.create<triton::BitcastOp>(
          loc, getElementTypeOrSelf(op.getResult().getType()),
          splatOp.getSrc());
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getResult().getType(),
                                                   rawPtr);
      return success();
    }
    return failure();
  }
  auto rawPtr = rewriter.create<triton::BitcastOp>(
      loc, getElementTypeOrSelf(op.getResult().getType()), info->ptr());
  auto newPtr =
      rewriter.create<triton::SplatOp>(loc, op.getResult().getType(), rawPtr)
          .getResult();
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newPtr.getType(), newPtr,
                                                info->offset());
  return success();
}

/// Check if argument of block has user by index.
static bool blockArgHasUse(Block *block, unsigned argIdx) {
  return !(block->getArgument(argIdx).use_empty());
}

/// Check if argument of blocks in region has user by index.
static bool regionArgHasUse(Region *region, unsigned argIdx) {
  for (auto &block : region->getBlocks()) {
    if (blockArgHasUse(&block, argIdx)) {
      return true;
    }
  }
  return false;
}

/// Check if argument of op has user in regions by index.
static bool opArghasUse(Operation *op, unsigned argIdx) {
  for (auto idx = 0; idx < op->getNumRegions(); idx++) {
    if (regionArgHasUse(&op->getRegion(idx), argIdx)) {
      return true;
    }
  }
  return false;
}

///
/// Canonicalization of pointers in Triton IR with controlflow ops.
///
/// The algorithm proceeds as follows:
/// - Step 1: Identify the target operation and match the target pointer
///   (e.g. those produced by AddPtrOp, MakeTensorPtrOp or AdvanceOp
///   and MakeTensorPtrOp)
/// - Step 2: Retrieve offset while simultaneously adding offset parameters
///   to the region and block.
/// - Step 3: Finally remove redundant pointer parameters and replace
///   the return values.
///
/// Ref: mlir/include/mlir/Interfaces/ControlFlowInterfaces.td.
///
template <typename OpT>
class PtrWithCFGStrengthReductionPattern
    : public OpInterfaceRewritePattern<OpT> {
public:
  explicit PtrWithCFGStrengthReductionPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern<OpT>(ctx) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    llvm_unreachable(
        "[PtrWithCFGStrengthReductionPattern] must override matchAndRewrite");
    return failure();
  }

private:
  bool isTritonPtrWithTensor(Type type) const {
    if (auto ptrType = type.dyn_cast<triton::PointerType>()) {
      return triton::getPointeeType(type).isa<TensorType>();
    }
    if (auto tensorTy = type.dyn_cast<TensorType>()) {
      return tensorTy.getElementType().isa<triton::PointerType>();
    }
    return false;
  }

  FailureOr<PtrInfo> getPreviousPtrInfo(Value ptr,
                                        PatternRewriter &rewriter) const {
    // Get previous tensor of ptr and offset.
    if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
      Value tracePtr = addPtrOp.getPtr();
      if (auto splatOp = tracePtr.getDefiningOp<triton::SplatOp>()) {
        return PtrInfo(splatOp.getOperand(), addPtrOp.getOffset());
      }
    }
    if (auto splatOp = ptr.getDefiningOp<triton::SplatOp>()) {
      rewriter.setInsertionPointAfter(splatOp);
      Type offsetType = rewriter.getIntegerType(32);
      assert(isTritonPtrWithTensor(splatOp.getResult().getType()));
      if (auto type = splatOp.getResult().getType().dyn_cast<ShapedType>()) {
        offsetType =
            RankedTensorType::get(type.getShape(), rewriter.getIntegerType(32));
      }
      auto offset = rewriter.create<arith::ConstantOp>(
          splatOp.getLoc(), rewriter.getZeroAttr(offsetType));
      return PtrInfo(splatOp.getOperand(), offset);
    }
    // Get previous ptr of tensor and infos.
    if (auto makeTensorPtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
      SmallVector<Value> sizes;
      SmallVector<Value> strides;
      SmallVector<Value> offsets;
      SmallVector<int32_t> orders;
      for (int i = 0; i < makeTensorPtrOp.getOffsets().size(); ++i) {
        sizes.push_back(makeTensorPtrOp.getShape()[i]);
        strides.push_back(makeTensorPtrOp.getStrides()[i]);
        offsets.push_back(makeTensorPtrOp.getOffsets()[i]);
        orders.push_back(makeTensorPtrOp.getOrder()[i]);
      }
      return PtrInfo(makeTensorPtrOp.getBase(), sizes, strides, offsets,
                     orders);
    }
    return failure();
  }

  FailureOr<Operation *>
  createOpWithNewResultTypes(PatternRewriter &rewriter, Operation *branchOp,
                             ValueRange valueRange = {},
                             TypeRange resultTypes = {}) const {
    return llvm::TypeSwitch<Operation *, FailureOr<Operation *>>(branchOp)
        .Case([&](scf::ForOp op) {
          auto newOp = rewriter.create<scf::ForOp>(
              op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
              valueRange);
          newOp.getRegion().takeBody(op.getRegion());
          return newOp;
        })
        .Case([&](scf::WhileOp op) {
          auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), resultTypes,
                                                     op.getInits());
          newOp.getBefore().takeBody(op.getBefore());
          newOp.getAfter().takeBody(op.getAfter());
          return newOp;
        })
        .Case([&](scf::IfOp op) {
          auto newOp = rewriter.create<scf::IfOp>(
              op.getLoc(), resultTypes, op.getCondition(),
              /* withElseRegion =  */ op.elseBlock() != nullptr);
          newOp.getThenRegion().takeBody(op.getThenRegion());
          newOp.getElseRegion().takeBody(op.getElseRegion());
          return newOp;
        })
        .Default([&](Operation *op) {
          return op->emitOpError("this operation is not supported!");
        });
  }

  bool isIntType(Value val, unsigned width) const {
    return getElementTypeOrSelf(val).isInteger(width);
  }

  void checkAndConvertOffsetToI64(const PtrInfo &info,
                                  PatternRewriter &rewriter) const {
    if (info.isBlockPtr()) {
      auto offsets = info.offsets();
      for (auto offset : offsets) {
        if (isIntType(offset, 32)) {
          rewriter.setInsertionPointAfter(offset.getDefiningOp());
          auto i64Offset = rewriter.create<arith::ExtSIOp>(
              offset.getLoc(), rewriter.getI64Type(), offset);
          offset.replaceAllUsesExcept(i64Offset, i64Offset);
        }
      }
    } else {
      auto offset = info.offset();
      if (isIntType(offset, 32)) {
        Type targetOffsetType = rewriter.getIntegerType(64);
        if (auto type = offset.getType().dyn_cast<ShapedType>()) {
          targetOffsetType = RankedTensorType::get(type.getShape(),
                                                   rewriter.getIntegerType(64));
        }
        rewriter.setInsertionPointAfter(offset.getDefiningOp());
        auto i64Offset = rewriter.create<arith::ExtSIOp>(
            offset.getLoc(), targetOffsetType, offset);
        offset.replaceAllUsesExcept(i64Offset, i64Offset);
      }
    }
  }

  bool checkAllOffsetTypeMatch(ArrayRef<PtrInfo> infoCache) const {
    if (infoCache.empty()) {
      return true;
    }
    auto firstInfo = infoCache[0];
    return llvm::all_of(infoCache, [&](const PtrInfo &info) {
      if (info.isBlockPtr()) {
        auto firstOffsets = firstInfo.offsets();
        auto offsets = info.offsets();
        assert(firstOffsets.size() == offsets.size());
        bool isAllSame = true;
        for (size_t i = 0; i < offsets.size(); i++) {
          if (firstOffsets[i].getType() != offsets[i].getType()) {
            isAllSame = false;
          }
        }
        return isAllSame;
      }
      return firstInfo.offset().getType() == info.offset().getType();
    });
  }

  LogicalResult processNonLoopRegionBranchOp(
      SmallVector<SuccessorInfo, 4> successorsInfoCache,
      RegionBranchOpInterface branchOp, PatternRewriter &rewriter) const {
    int argSize = 0;
    if (!successorsInfoCache.empty()) {
      for (auto &block : successorsInfoCache[0].region->getBlocks()) {
        auto *terminatorOp = block.getTerminator();
        assert(terminatorOp);
        if (terminatorOp->hasTrait<OpTrait::ReturnLike>()) {
          argSize = terminatorOp->getNumOperands();
        }
        break;
      }
    }
    // Process ptr with different offset type.
    for (int argIdx = 0; argIdx < argSize; argIdx++) {
      SmallVector<PtrInfo, 4> predecessorsPtrInfoCache;
      for (auto &info : successorsInfoCache) {
        for (auto &block : info.region->getBlocks()) {
          auto *terminatorOp = block.getTerminator();
          assert(terminatorOp);
          Value argVal = terminatorOp->getOperand(argIdx);
          if (isTritonPtrWithTensor(argVal.getType())) {
            auto info = getPreviousPtrInfo(argVal, rewriter);
            if (!failed(info)) {
              predecessorsPtrInfoCache.push_back(*info);
            }
          }
        }
      }
      if (!checkAllOffsetTypeMatch(predecessorsPtrInfoCache)) {
        bool converted = false;
        for (auto &info : predecessorsPtrInfoCache) {
          converted = true;
          checkAndConvertOffsetToI64(info, rewriter);
        }
        if (converted) {
          return success();
        }
      }
    }

    bool isChanged = false;
    for (int argIdx = 0; argIdx < argSize; argIdx++) {
      bool allPtrReplacable = true;
      SmallVector<PtrInfo, 4> predecessorsPtrInfoCache;
      SmallVector<Operation *, 4> terminatorOpsCache;
      // Check all regions and blocks.
      for (auto &info : successorsInfoCache) {
        // Loop to check all regions.
        for (auto &block : info.region->getBlocks()) {
          auto *terminatorOp = block.getTerminator();
          assert(terminatorOp);
          Value argVal = terminatorOp->getOperand(argIdx);
          if (isTritonPtrWithTensor(argVal.getType())) {
            auto info = getPreviousPtrInfo(argVal, rewriter);
            if (!failed(info)) {
              predecessorsPtrInfoCache.push_back(*info);
              terminatorOpsCache.push_back(terminatorOp);
              continue;
            }
          }
          allPtrReplacable = false;
        }
      }
      if (!allPtrReplacable || terminatorOpsCache.empty()) {
        continue;
      }
      // Replace all ptr.
      auto firstInfo = predecessorsPtrInfoCache[0];
      bool sameRawPtr = true;
      for (unsigned idx = 1; idx < terminatorOpsCache.size(); idx++) {
        auto info = predecessorsPtrInfoCache[idx];
        if (info.ptr() != firstInfo.ptr()) {
          sameRawPtr = false;
        }
      }
      for (unsigned idx = 0; idx < terminatorOpsCache.size(); idx++) {
        auto info = predecessorsPtrInfoCache[idx];
        auto *terminatorOp = terminatorOpsCache[idx];
        SmallVector<Value> packedInfo;
        if (!sameRawPtr) {
          packedInfo = packPtrInfo(info);
        } else {
          packedInfo = packPtrInfo(info, /* ignorePtr = */ true);
        }
        terminatorOp->insertOperands(argSize, packedInfo);
        terminatorOp->eraseOperand(argIdx);
      }
      auto resultTypes = terminatorOpsCache[0]->getOperandTypes();
      auto newBranchOp =
          createOpWithNewResultTypes(rewriter, branchOp, {}, resultTypes);
      assert(!failed(newBranchOp));

      // Reconstruct splat and addptr outside the parent op.
      auto rawPtr = branchOp->getResult(argIdx);
      reconstructAfterOp(*newBranchOp, rawPtr,
                         /* startIdx = */ argSize - 1,
                         /* hasPtr = */ !sameRawPtr, firstInfo, rewriter);
      // Replace all results for old results.
      for (int idx = 0; idx < argSize; idx++) {
        if (idx < argIdx) {
          rewriter.replaceAllUsesWith(branchOp->getOpResult(idx),
                                      (*newBranchOp)->getOpResult(idx));
        } else if (idx > argIdx) {
          rewriter.replaceAllUsesWith(branchOp->getOpResult(idx),
                                      (*newBranchOp)->getOpResult(idx - 1));
        }
      }
      branchOp = cast<RegionBranchOpInterface>(*newBranchOp);
      isChanged = true;
    }
    if (isChanged) {
      return success();
    }
    return failure();
  }

  LogicalResult
  processLoopWithSingleRegionBranch(SuccessorInfo &mainInfo,
                                    RegionBranchOpInterface branchOp,
                                    PatternRewriter &rewriter) const {
    unsigned iterSize = mainInfo.getResultsSize();
    unsigned blockArgSize = mainInfo.getArgumentSize();
    unsigned numOperands = branchOp->getNumOperands();

    // Process input operands.
    for (unsigned resultIdx = 0; resultIdx < iterSize; resultIdx++) {
      unsigned operandIdx = numOperands - iterSize + resultIdx;
      Value argVal = branchOp->getOperand(operandIdx);
      if (!isTritonPtrWithTensor(argVal.getType())) {
        continue;
      }
      auto info = getPreviousPtrInfo(argVal, rewriter);
      unsigned argIdx = resultIdx + blockArgSize - iterSize;
      if (!failed(info) && opArghasUse(branchOp, argIdx)) {
        // Add new offset operand after argVal.
        SmallVector<Value, 4> newOperands;
        for (unsigned idx = 0; idx < iterSize; idx++) {
          unsigned tempIdx = numOperands - iterSize + idx;
          newOperands.push_back(branchOp->getOperand(tempIdx));
          if (resultIdx == idx) {
            auto packedInfo = packPtrInfo(*info, /* ignorePtr = */ true);
            for (auto val : packedInfo) {
              newOperands.push_back(val);
            }
          }
        }
        auto newBranchOp =
            createOpWithNewResultTypes(rewriter, branchOp, newOperands);
        assert(!failed(newBranchOp));

        unsigned stepSize = 0;
        for (auto &block : (*newBranchOp)->getRegion(0).getBlocks()) {
          auto rets = reconstructPtrInBlock(&block, argIdx, argIdx + 1, *info,
                                            rewriter, /*hasPtr=*/false);
          stepSize = rets.size();
          auto terminatorOp = block.getTerminator();
          assert(terminatorOp);
          terminatorOp->insertOperands(resultIdx + 1, rets);
        }
        for (unsigned retIdx = 0; retIdx < branchOp->getNumResults();
             retIdx++) {
          if (retIdx < argIdx) {
            rewriter.replaceAllUsesWith(branchOp->getOpResult(retIdx),
                                        (*newBranchOp)->getOpResult(retIdx));
          } else {
            rewriter.replaceAllUsesWith(
                branchOp->getOpResult(retIdx),
                (*newBranchOp)->getOpResult(retIdx + stepSize));
          }
        }
        rewriter.eraseOp(branchOp);
        return success();
      }
    }

    // Process output results.
    bool isOutsChanged = false;
    for (unsigned resultIdx = 0; resultIdx < iterSize; resultIdx++) {
      unsigned argIdx = resultIdx + blockArgSize - iterSize;
      SmallVector<PtrInfo, 4> ptrInfoCache;
      bool replacable = true;
      for (auto &block : mainInfo.region->getBlocks()) {
        auto *terminatorOp = block.getTerminator();
        assert(terminatorOp);
        Value iterVal = terminatorOp->getOperand(resultIdx);
        if (isTritonPtrWithTensor(iterVal.getType())) {
          auto info = getPreviousPtrInfo(iterVal, rewriter);
          if (!failed(info) && !opArghasUse(branchOp, argIdx) // Check no use.
          ) {
            auto len = packPtrInfo(*info, /* ignorePtr = */ true).size();
            // Check offset.
            if (verifyArgsMatchTerminatorInputsInBlock(&block, argIdx + 1,
                                                       resultIdx + 1, len)) {
              ptrInfoCache.push_back(*info);
              continue;
            }
          }
        }
        replacable = false;
      }
      if (replacable && !ptrInfoCache.empty()) {
        // Remove redundant operand and result.
        unsigned packedInfoSize = 0;
        unsigned cacheIdx = 0;
        for (auto &block : branchOp->getRegion(0).getBlocks()) {
          auto *terminatorOp = block.getTerminator();
          assert(terminatorOp);
          auto info = ptrInfoCache[cacheIdx++];
          block.eraseArgument(argIdx);
          auto packedInfo = packPtrInfo(info, /* ignorePtr = */ true);
          packedInfoSize = packedInfo.size();
          terminatorOp->setOperands(resultIdx + 1, packedInfoSize, packedInfo);
          terminatorOp->eraseOperand(resultIdx);
        }

        SmallVector<Value, 4> newOperands;
        for (unsigned idx = 0; idx < iterSize; idx++) {
          unsigned tempIdx = numOperands - iterSize + idx;
          if (resultIdx != idx) {
            newOperands.push_back(branchOp->getOperand(tempIdx));
          }
        }

        auto newBranchOp =
            createOpWithNewResultTypes(rewriter, branchOp, newOperands);
        assert(!failed(newBranchOp));

        auto info = ptrInfoCache[0];
        auto rawPtr = branchOp->getResult(resultIdx);
        reconstructAfterOp(*newBranchOp, rawPtr,
                           /* startIdx = */ resultIdx,
                           /* hasPtr = */ false, info, rewriter);

        for (unsigned retIdx = 0; retIdx < branchOp->getNumResults();
             retIdx++) {
          if (retIdx < resultIdx) {
            rewriter.replaceAllUsesWith(branchOp->getOpResult(retIdx),
                                        (*newBranchOp)->getOpResult(retIdx));
          } else if (retIdx > resultIdx) {
            rewriter.replaceAllUsesWith(
                branchOp->getOpResult(retIdx),
                (*newBranchOp)->getOpResult(retIdx - 1));
          }
        }
        branchOp = cast<RegionBranchOpInterface>(*newBranchOp);
        isOutsChanged = true;
      }
    }
    if (isOutsChanged) {
      return success();
    }
    return failure();
  }

  LogicalResult
  processLoopWithMultiRegionBranch(RegionBranchOpInterface branchOp,
                                   PatternRewriter &rewriter) const {
    return llvm::TypeSwitch<Operation *, LogicalResult>(branchOp)
        .Case([&](scf::WhileOp op) {
          return processSpecialOperation<scf::WhileOp>(op, rewriter);
        })
        .Default([&](Operation *op) { return failure(); });
  }

  template <typename OperationT>
  LogicalResult processSpecialOperation(OperationT op,
                                        PatternRewriter &rewriter) const {
    return failure();
  }

  LogicalResult
  processLoopRegionBranchOp(SmallVector<SuccessorInfo, 4> successorsInfoCache,
                            RegionBranchOpInterface branchOp,
                            PatternRewriter &rewriter) const {
    int loopInfoIdx = -1;
    for (size_t idx = 0; idx < successorsInfoCache.size(); idx++) {
      if (llvm::any_of(
              successorsInfoCache[idx].successors,
              [](const RegionSuccessor &succ) { return succ.isParent(); })) {
        loopInfoIdx = idx;
        break;
      }
    }
    assert(loopInfoIdx >= 0);

    SuccessorInfo mainInfo = successorsInfoCache[loopInfoIdx];
    if (successorsInfoCache.size() == 1) {
      // Process loop like op with only one region branch.
      return processLoopWithSingleRegionBranch(mainInfo, branchOp, rewriter);
    }
    // For loop like op with multi-region branchs.
    return processLoopWithMultiRegionBranch(branchOp, rewriter);
  }

  bool needClone(BranchOpInterface branchOp, PatternRewriter &rewriter) const {
    for (unsigned idx = 0; idx < branchOp->getNumSuccessors(); idx++) {
      Block *block = branchOp->getSuccessor(idx);
      for (unsigned argIdx = 0; argIdx < block->getNumArguments(); argIdx++) {
        // Check all predecessors branch Op for addArgument index equal to
        // 'argIdx'.
        if (!blockArgHasUse(block, argIdx)) {
          continue;
        }
        bool allPtrReplacable = true;
        SmallVector<PtrInfo, 4> predecessorsPtrInfoCache;
        SmallVector<SuccessorOperands, 4> predecessorSucOperandsCache;
        for (auto it = block->pred_begin(), e = block->pred_end(); it != e;
             ++it) {
          auto branch = cast<BranchOpInterface>((*it)->getTerminator());
          unsigned index = it.getSuccessorIndex();

          SuccessorOperands sucOperands = branch.getSuccessorOperands(index);
          predecessorSucOperandsCache.push_back(sucOperands);
          Value argVal = sucOperands[argIdx];
          if (isTritonPtrWithTensor(argVal.getType())) {
            auto info = getPreviousPtrInfo(argVal, rewriter);
            if (!failed(info)) {
              predecessorsPtrInfoCache.push_back(*info);
              continue;
            }
          }
          allPtrReplacable = false;
        }

        // Process ptr with different offset type.
        if (!checkAllOffsetTypeMatch(predecessorsPtrInfoCache)) {
          return false;
        }

        // Reconstruct all results.
        if (allPtrReplacable && (!predecessorsPtrInfoCache.empty())) {
          return true;
        }
      }
    }
    return false;
  }
};

/// Clone block.
static Block *cloneBlock(Block *block, PatternRewriter &rewriter) {
  Block *newBlock = rewriter.createBlock(block->getParent());
  IRMapping valueMap;
  for (auto arg : block->getArguments()) {
    auto clonedArg = newBlock->addArgument(arg.getType(), arg.getLoc());
    valueMap.map(arg, clonedArg);
  }
  // rewriter.setInsertionPointToEnd(newBlock);
  for (Operation &op : *block) {
    rewriter.clone(op, valueMap);
  }
  return newBlock;
}

/// Canonicalization for operations with 'BranchOpInterface'.
/// * cf.br
/// * cf.cond_br
/// * cf.switch
template <>
LogicalResult
PtrWithCFGStrengthReductionPattern<BranchOpInterface>::matchAndRewrite(
    BranchOpInterface branchOp, PatternRewriter &rewriter) const {
  DenseSet<Block *> blockSets;
  // Clone block when encounter duplicated block.
  bool isNeedClone = needClone(branchOp, rewriter);
  if (isNeedClone) {
    for (unsigned i = 0; i < branchOp->getNumSuccessors(); i++) {
      Block *sucBlock = branchOp->getSuccessor(i);
      if (!blockSets.contains(sucBlock)) {
        blockSets.insert(sucBlock);
        continue;
      }
      Block *clonedBlock = cloneBlock(sucBlock, rewriter);
      branchOp->setSuccessor(clonedBlock, i);
    }
  }

  bool replaced = false;
  for (unsigned idx = 0; idx < branchOp->getNumSuccessors(); idx++) {
    Block *block = branchOp->getSuccessor(idx);
    for (unsigned argIdx = 0; argIdx < block->getNumArguments(); argIdx++) {
      // Check all predecessors branch Op for addArgument index equal to
      // 'argIdx'.
      if (!blockArgHasUse(block, argIdx)) {
        continue;
      }
      bool allPtrReplacable = true;
      SmallVector<PtrInfo, 4> predecessorsPtrInfoCache;
      SmallVector<SuccessorOperands, 4> predecessorSucOperandsCache;
      for (auto it = block->pred_begin(), e = block->pred_end(); it != e;
           ++it) {
        auto branch = cast<BranchOpInterface>((*it)->getTerminator());
        unsigned index = it.getSuccessorIndex();

        SuccessorOperands sucOperands = branch.getSuccessorOperands(index);
        predecessorSucOperandsCache.push_back(sucOperands);
        Value argVal = sucOperands[argIdx];
        if (isTritonPtrWithTensor(argVal.getType())) {
          auto info = getPreviousPtrInfo(argVal, rewriter);
          if (!failed(info)) {
            predecessorsPtrInfoCache.push_back(*info);
            continue;
          }
        }
        allPtrReplacable = false;
      }
      // Process ptr with different offset type.
      if (!checkAllOffsetTypeMatch(predecessorsPtrInfoCache)) {
        bool converted = false;
        for (auto &info : predecessorsPtrInfoCache) {
          converted = true;
          checkAndConvertOffsetToI64(info, rewriter);
        }
        if (converted) {
          return success();
        }
      }

      // Reconstruct all results.
      if (allPtrReplacable && (!predecessorsPtrInfoCache.empty())) {
        // Rewrite target block.
        auto info = predecessorsPtrInfoCache[0];
        reconstructPtrInBlock(block, argIdx, block->getNumArguments(), info,
                              rewriter);
        block->eraseArgument(argIdx);
        // Rewrite all predecessor branchOp.
        for (unsigned i = 0; i < predecessorsPtrInfoCache.size(); i++) {
          info = predecessorsPtrInfoCache[i];
          predecessorSucOperandsCache[i].append(packPtrInfo(info));
          predecessorSucOperandsCache[i].erase(argIdx);
        }
        replaced = true;
      }
    }
  }
  if (replaced) {
    return success();
  }
  if (isNeedClone) {
    return success();
  }
  return failure();
}

/// Canonicalization for operations with 'RegionBranchOpInterface'.
/// * scf.for
/// * scf.if
/// * scf.while
template <>
LogicalResult
PtrWithCFGStrengthReductionPattern<RegionBranchOpInterface>::matchAndRewrite(
    RegionBranchOpInterface branchOp, PatternRewriter &rewriter) const {
  SmallVector<SuccessorInfo, 4> successorsInfoCache;
  for (int i = 0, e = branchOp->getNumRegions(); i != e; ++i) {
    SmallVector<RegionSuccessor, 2> successors;
    auto *currRegion = &branchOp->getRegion(i);
    branchOp.getSuccessorRegions(currRegion, successors);
    successorsInfoCache.push_back(SuccessorInfo{
        successors, &branchOp->getRegion(i), branchOp.isRepetitiveRegion(i)});
  }
  bool hasLoopLike =
      llvm::any_of(successorsInfoCache,
                   [](const SuccessorInfo &info) { return info.isLoopLike; });
  bool hasNonLoopLike =
      llvm::any_of(successorsInfoCache,
                   [](const SuccessorInfo &info) { return !info.isLoopLike; });
  if (hasLoopLike && hasNonLoopLike) {
    // can't support RegionBranchOp with both loop and non-loop region.
    return failure();
  }

  // With loop semantic.
  if (hasLoopLike) {
    LLVM_DEBUG(llvm::dbgs() << "[RegionBranchOpInterface] Deal with loop: \n";);
    return processLoopRegionBranchOp(successorsInfoCache, branchOp, rewriter);
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[RegionBranchOpInterface] Deal with non-loop: \n";);
  // Without loop semantic.
  return processNonLoopRegionBranchOp(successorsInfoCache, branchOp, rewriter);
}

/// Process special case: scf::WhileOp.
template <>
template <>
LogicalResult PtrWithCFGStrengthReductionPattern<RegionBranchOpInterface>::
    processSpecialOperation(scf::WhileOp op, PatternRewriter &rewriter) const {
  for (unsigned initIter = 0; initIter < op.getInits().size(); initIter++) {
    Value argVal = op->getOperand(initIter);
    if (isTritonPtrWithTensor(argVal.getType())) {
      auto info = getPreviousPtrInfo(argVal, rewriter);
      if (!failed(info) && regionArgHasUse(&op.getBefore(), initIter)) {
        auto packedInfo = packPtrInfo(*info, /* ignorePtr = */ true);
        op->insertOperands(initIter + 1, packedInfo);
        // Change before region.
        auto &blockBefore = op.getBefore().front();
        auto rets =
            reconstructPtrInBlock(&blockBefore, initIter, initIter + 1, *info,
                                  rewriter, /* hasPtr = */ false);
        // Change after region.
        auto yieldOp = op.getYieldOp();
        yieldOp->insertOperands(initIter + 1, rets);
        return success();
      }
      if (!failed(info) && !regionArgHasUse(&op.getBefore(), initIter)) {
        // Change after region.
        auto yieldOp = op.getYieldOp();
        auto yieldInfo =
            getPreviousPtrInfo(yieldOp->getOperand(initIter), rewriter);
        if (!failed(yieldInfo)) {
          auto packedInfo = packPtrInfo(*yieldInfo, /* ignorePtr = */ true);
          yieldOp->eraseOperands(initIter, 1 + packedInfo.size());
          yieldOp->insertOperands(initIter, packedInfo);
          // Remove ptr in before block and operands.
          op->eraseOperand(initIter);
          auto &blockBefore = op.getBefore().front();
          blockBefore.eraseArgument(initIter);
          return success();
        }
      }
    }
  }
  // Process before region.
  auto condOp = op.getConditionOp();
  for (unsigned argIter = 1; argIter < condOp->getNumOperands(); argIter++) {
    Value argVal = condOp->getOperand(argIter);
    if (isTritonPtrWithTensor(argVal.getType())) {
      auto info = getPreviousPtrInfo(argVal, rewriter);
      if (!failed(info)) {
        // Remove redundant operand and result.
        SmallVector<Type, 4> newRetTypes;
        auto packedInfo = packPtrInfo(*info, /* ignorePtr = */ true);
        for (unsigned idx = 1; idx < condOp->getNumOperands(); idx++) {
          if (argIter == idx) {
            for (auto val : packedInfo) {
              newRetTypes.push_back(val.getType());
            }
          } else {
            newRetTypes.push_back(condOp->getOperandTypes()[idx]);
          }
        }
        auto newOp = createOpWithNewResultTypes(rewriter, op, {}, newRetTypes);
        assert(!failed(newOp));

        auto newWhileOp = cast<scf::WhileOp>(*newOp);
        auto newCondOp = newWhileOp.getConditionOp();
        newCondOp->insertOperands(argIter + 1, packedInfo);
        newCondOp->eraseOperand(argIter);
        // Process after region.
        unsigned bArgIdx = argIter - 1; // Minus first cond arg.
        auto &afterBlock = newWhileOp.getAfter().front();
        auto rets =
            reconstructPtrInBlock(&afterBlock, bArgIdx, bArgIdx + 1, *info,
                                  rewriter, /* hasPtr = */ false);
        afterBlock.eraseArgument(bArgIdx);
        // Reconstruct output ptr after op.
        auto rawRetPtr = op->getOpResult(bArgIdx);
        reconstructAfterOp(newWhileOp, rawRetPtr,
                           /* startIdx = */ bArgIdx,
                           /* hasPtr = */ false, *info, rewriter);
        // Replace all uses after while op.
        for (unsigned retIdx = 0; retIdx < op->getNumResults(); retIdx++) {
          if (retIdx < bArgIdx) {
            rewriter.replaceAllUsesWith(op->getOpResult(retIdx),
                                        newWhileOp->getOpResult(retIdx));
          } else {
            rewriter.replaceAllUsesWith(
                op->getOpResult(retIdx),
                newWhileOp->getOpResult(retIdx + rets.size() - 1));
          }
        }
        rewriter.eraseOp(op);
        return success();
      }
    }
  }
  return failure();
}

template <typename... OpTs> class RegisterPatterns;

template <typename OpT, typename... OpTs>
class RegisterPatterns<OpT, OpTs...> : RegisterPatterns<OpTs...> {
public:
  explicit RegisterPatterns(RewritePatternSet &set, MLIRContext *ctx)
      : RegisterPatterns<OpTs...>(set, ctx) {
    set.insert<PtrStrengthReductionPattern<OpT>>(ctx);
  }
};

template <> class RegisterPatterns<> {
public:
  explicit RegisterPatterns(RewritePatternSet &set, MLIRContext *ctx) {
    // Empty register.
  }
};

struct PointerStrengthReductionPtrPass
    : public PointerStrengthReductionPtrBase<PointerStrengthReductionPtrPass> {

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patternsNormal(&ctx);
    RegisterPatterns<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
        >(patternsNormal, &ctx);
    patternsNormal
        .insert<PtrWithCFGStrengthReductionPattern<BranchOpInterface>>(&ctx);
    patternsNormal
        .insert<PtrWithCFGStrengthReductionPattern<RegionBranchOpInterface>>(
            &ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patternsNormal)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
std::unique_ptr<Pass> createPointerStrengthReductionPass() {
  return std::make_unique<PointerStrengthReductionPtrPass>();
}

} // namespace triton
} // namespace mlir
