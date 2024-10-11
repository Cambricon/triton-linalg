//===- PointerMetaInfoTracker.cpp - Trace the pointer pattern --*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/Triton/Utils/PointerMetaInfoTracker.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include <stddef.h>

namespace mlir {
class Operation;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

template <>
LogicalResult TensorPointerMetaInfoTracker::parseOp<triton::MakeTensorPtrOp>(
    triton::MakeTensorPtrOp op, Location loc,
    ConversionPatternRewriter &rewriter) {
  this->base = op.getBase();
  this->order = op.getOrder();
  size_t size = op.getOffsets().size();
  // Cast offsets/sizes/strides into index.
  for (auto i : llvm::seq<size_t>(0, size)) {
    this->offsets.push_back(rewriter.createOrFold<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getOffsets()[i]));
    this->sizes.push_back(rewriter.createOrFold<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getShape()[i]));
    this->strides.push_back(rewriter.createOrFold<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getStrides()[i]));
  }

  return success();
}

template <>
LogicalResult TensorPointerMetaInfoTracker::parseOp<triton::AdvanceOp>(
    triton::AdvanceOp op, Location loc, ConversionPatternRewriter &rewriter) {
  if (failed(parse(op.getPtr(), loc, rewriter)))
    return failure();

  for (size_t i = 0; i < getRank(); ++i) {
    this->offsets[i] = rewriter.createOrFold<arith::AddIOp>(
        loc,
        rewriter.createOrFold<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                  op.getOffsets()[i]),
        getValueOrCreateConstantIndexOp(rewriter, loc, this->offsets[i]));
  }

  return success();
}

LogicalResult
TensorPointerMetaInfoTracker::parse(Value operand, Location loc,
                                    ConversionPatternRewriter &rewriter) {
  auto *defOp = operand.getDefiningOp();
  if (!defOp) {
    return rewriter.notifyMatchFailure(loc,
                                       "Unsupported case for block argument");
  }

  return llvm::TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case<triton::MakeTensorPtrOp, triton::AdvanceOp>(
          [&](auto op) { return parseOp(op, loc, rewriter); })
      .Default([](Operation *) { return failure(); });
}

///////////////////////// PointerMetaInfoTracker /////////////////////////

template <>
LogicalResult PointerMetaInfoTracker::parseOp<triton::AddPtrOp>(
    triton::AddPtrOp op, Location loc, ConversionPatternRewriter &rewriter) {
  if (failed(parse(op.getPtr(), loc, rewriter)))
    return failure();

  Value currentOffset = op.getOffset();
  Type offsetElementType = getElementTypeOrSelf(this->offset.getType());
  Type currentOffsetElementType = getElementTypeOrSelf(currentOffset.getType());

  // Promote offset type to the type of largest bitwith.
  if (offsetElementType.getIntOrFloatBitWidth() >
      currentOffsetElementType.getIntOrFloatBitWidth()) {
    currentOffset = rewriter.createOrFold<arith::ExtSIOp>(
        loc, this->offset.getType(), currentOffset);
  } else if (offsetElementType.getIntOrFloatBitWidth() <
             currentOffsetElementType.getIntOrFloatBitWidth()) {
    this->offset = rewriter.createOrFold<arith::ExtSIOp>(
        loc, currentOffset.getType(), this->offset);
  }

  this->offset =
      rewriter.createOrFold<arith::AddIOp>(loc, currentOffset, this->offset);
  return success();
}

template <>
LogicalResult PointerMetaInfoTracker::parseOp<triton::BitcastOp>(
    triton::BitcastOp op, Location loc, ConversionPatternRewriter &rewriter) {
  // In order to address the i64 performance issue when tracking
  // offset computations, the offset address is not multiplied by the actual
  // bytes of the element it points to, so currently only support cast between
  // types of the same bitwidth or conversion between tt.ptr<i1> and tt.ptr<i8>.
  unsigned srcPointeeBitWidth =
      triton::getPointeeBitWidth(op.getOperand().getType());
  unsigned dstPointeeBitWidth =
      triton::getPointeeBitWidth(op.getResult().getType());

  bool valid = srcPointeeBitWidth == dstPointeeBitWidth;
  valid |= (srcPointeeBitWidth == 1 && dstPointeeBitWidth == 8);
  valid |= (srcPointeeBitWidth == 8 && dstPointeeBitWidth == 1);
  if (!valid)
    return failure();
  if (failed(parse(op.getOperand(), loc, rewriter)))
    return failure();
  return success();
}

template <>
LogicalResult PointerMetaInfoTracker::parseOp<triton::SplatOp>(
    triton::SplatOp op, Location loc, ConversionPatternRewriter &rewriter) {
  if (failed(parse(op.getOperand(), loc, rewriter)))
    return failure();
  this->offset = rewriter.create<triton::SplatOp>(
      loc,
      RankedTensorType::get(
          op.getResult().getType().cast<ShapedType>().getShape(),
          this->offset.getType()),
      this->offset);
  return success();
}

template <>
LogicalResult PointerMetaInfoTracker::parseOp<triton::ExpandDimsOp>(
    triton::ExpandDimsOp op, Location loc,
    ConversionPatternRewriter &rewriter) {
  if (failed(parse(op.getOperand(), loc, rewriter)))
    return failure();
  this->offset =
      rewriter.create<triton::ExpandDimsOp>(loc, this->offset, op.getAxis());
  return success();
}

template <>
LogicalResult PointerMetaInfoTracker::parseOp<triton::BroadcastOp>(
    triton::BroadcastOp op, Location loc, ConversionPatternRewriter &rewriter) {
  if (failed(parse(op.getOperand(), loc, rewriter)))
    return failure();
  this->offset = rewriter.create<triton::BroadcastOp>(
      loc,
      RankedTensorType::get(
          op.getResult().getType().cast<ShapedType>().getShape(),
          getElementTypeOrSelf(this->offset.getType())),
      this->offset);
  return success();
}

FailureOr<bool>
PointerMetaInfoTracker::parse(Value operand, Location loc,
                              ConversionPatternRewriter &rewriter) {
  auto *defOp = operand.getDefiningOp();
  // FIXME: Currently, for operators on the whitelist, further
  // tracking is conducted back to the end of the function's basic block
  // arguments, choosing to serve as the base of the pointer. This is mainly
  // done to avoid issues with unsupported negative indexing in instructions.
  // However, in some scenarios, such as those involving control flows where
  // different branches yield different pointers, it's not possible to continue
  // tracking the base pointer. Therefore, the issue of negative indexing cannot
  // be resolved scientifically. A compromise will be made in the future, which
  // is to mark the instructions when tracking is possible and normally
  // decrement the instruction level; otherwise, when generating IR1
  // instructions later, the negative index values will be additionally removed.
  // This is a temporary solution, and such an approach will be implemented in
  // the next version.
  if (defOp) {
    // We only process operators that are not added to the whitelist. If an
    // operator is processed by the whitelist but returns a failure, we also
    // mark it as a failure. For example, tt.bitwise for cases with unequal bit
    // width, so here we use isProcessedSuccessfully to store its state.
    bool isProcessedSuccessfully = true;
    auto res =
        llvm::TypeSwitch<Operation *, LogicalResult>(defOp)
            .Case<triton::AddPtrOp, triton::BitcastOp, triton::SplatOp,
                  triton::BroadcastOp, triton::ExpandDimsOp>([&](auto op) {
              auto ret = parseOp(op, loc, rewriter);
              isProcessedSuccessfully = ret.succeeded();
              return ret;
            })
            .Default([](Operation *) { return failure(); });
    if (res.succeeded())
      return isProcessedSuccessfully;
    if (res.failed() && !isProcessedSuccessfully)
      return failure(); // res
  }
  if (!operand.getType().isa<triton::PointerType>())
    return rewriter.notifyMatchFailure(
        loc, "only support base ptr of triton scalar pointer");
  this->base = operand;
  this->offset = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  return true;
}
