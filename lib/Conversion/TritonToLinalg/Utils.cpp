//===- Utils.cpp - Triton to Linalg utils impl ------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <optional>

#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Utils/ArithUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir {
class Operation;
class MLIRContext;
} // namespace mlir

using namespace mlir;

Value mlir::triton::getPadOrInsertOpWithOther(Location loc, Value other,
                                               Type otherType, Value source,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes,
                                               OpBuilder &rewriter) {
  auto otherShapedType = otherType.cast<ShapedType>();
  assert(otherShapedType.hasStaticShape() && "other val shape must be static.");
  Type elementType = otherShapedType.getElementType();
  auto rank = otherShapedType.getRank();
  // For rank = 0 tensor value, return itself.
  if (rank == 0)
    return source;

  do {
    // If other val is not from operation, fallback to create insert_slice op.
    if (other && !other.getDefiningOp()) {
      break;
    }
    Value pValue;
    if (other) {
      // Extract pad init value by analysing other defining op.
      FailureOr<Value> res =
          llvm::TypeSwitch<Operation *, FailureOr<Value>>(other.getDefiningOp())
              .Case<arith::ConstantOp, arith::ConstantIntOp,
                    arith::ConstantFloatOp>([&](auto constOp) {
                return mlir::triton::getSplatValue(rewriter, constOp);
              })
              .Case<linalg::FillOp>(
                  [&](linalg::FillOp fillOp) { return fillOp.value(); })
              .Default([&](Operation *op) { return failure(); });
      // Fail to get pad init value, fallback to create insert_slice op.
      if (failed(res)) {
        break;
      }
      pValue = *res;
    } else {
      // Other val does not exist.
      // Use c0 as pad value, as no other specific means a unintialized value.
      pValue = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getZeroAttr(elementType));
    }

    auto padInit = rewriter.create<tensor::EmptyOp>(
        loc, otherShapedType.getShape(), elementType);
    SmallVector<Value> lowPads, highPads;
    for (auto i = 0; i < rank; i++) {
      auto index = rewriter.create<arith::ConstantIndexOp>(loc, i);
      auto dstDim = rewriter.create<tensor::DimOp>(loc, padInit, index);
      // Get low pad val.
      auto lowPad = getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i]);
      lowPads.push_back(lowPad);
      // Get high pad val.
      auto highOffset = rewriter.create<arith::AddIOp>(
          loc, lowPad,
          getValueOrCreateConstantIndexOp(rewriter, loc, sizes[i]));
      highPads.push_back(
          rewriter.create<arith::SubIOp>(loc, dstDim, highOffset));
    }
    return rewriter
        .create<mlir::triton::linalg_ext::PadOp>(loc, source, padInit, pValue,
                                                  lowPads, highPads)
        .getResult()[0];
  } while (false);

  return rewriter
      .create<tensor::InsertSliceOp>(
          loc, source, other,
          /*offsets=*/offsets,
          /*sizes=*/sizes,
          /*strides=*/
          SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)))
      .getResult();
}

StringAttr mlir::triton::getCacheModeAttr(MLIRContext *context,
                                           triton::CacheModifier mode) {
  switch (mode) {
  case triton::CacheModifier::CA:
  case triton::CacheModifier::CG:
    return StringAttr::get(context, "cmnormal");
  case triton::CacheModifier::WB:
  case triton::CacheModifier::CS:
  case triton::CacheModifier::WT:
    return StringAttr::get(context, "cmtransient");
  default:
    return nullptr;
  }
}

FailureOr<triton::linalg_ext::MemoryOrder>
mlir::triton::getLinalgExtAtomicMemoryOrder(triton::MemSemantic memSem) {
  switch (memSem) {
  case triton::MemSemantic::RELAXED:
    return triton::linalg_ext::MemoryOrder::relaxed;
  case triton::MemSemantic::ACQUIRE:
    return triton::linalg_ext::MemoryOrder::acquire;
  case triton::MemSemantic::RELEASE:
    return triton::linalg_ext::MemoryOrder::release;
  case triton::MemSemantic::ACQUIRE_RELEASE:
    return triton::linalg_ext::MemoryOrder::acq_rel;
  default:
    llvm_unreachable("Invalid MemoryOrder");
    return failure();
  }
}
