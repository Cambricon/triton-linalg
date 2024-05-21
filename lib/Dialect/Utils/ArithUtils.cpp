//===- ArithUtils.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include "triton-linalg/Dialect/Utils/ArithUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Dialect/Arith/Transforms/EmulateWideInt.cpp
//===----------------------------------------------------------------------===//
Value mlir::triton::createScalarOrSplatConstant(OpBuilder &builder,
                                                 Location loc, Type type,
                                                 const APInt &value) {
  TypedAttr attr;
  if (isa<IntegerType>(type)) {
    attr = builder.getIntegerAttr(type, value);
  } else {
    auto vecTy = cast<ShapedType>(type);
    attr = SplatElementsAttr::get(vecTy, value);
  }

  return builder.create<arith::ConstantOp>(loc, attr);
}

Value mlir::triton::createScalarOrSplatConstant(OpBuilder &builder,
                                                 Location loc, Type type,
                                                 int64_t value) {
  unsigned elementBitWidth = 0;
  if (auto intTy = dyn_cast<IntegerType>(type))
    elementBitWidth = intTy.getWidth();
  else
    elementBitWidth = cast<ShapedType>(type).getElementTypeBitWidth();

  return createScalarOrSplatConstant(builder, loc, type,
                                     APInt(elementBitWidth, value));
}

Value mlir::triton::createScalarOrSplatConstant(OpBuilder &builder,
                                                 Location loc, Type type,
                                                 const APFloat &value) {
  if (isa<FloatType>(type))
    return builder.createOrFold<arith::ConstantOp>(
        loc, type, builder.getFloatAttr(type, value));
  TypedAttr splat = SplatElementsAttr::get(cast<ShapedType>(type), value);
  return builder.createOrFold<arith::ConstantOp>(loc, type, splat);
}
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Dialect/Arith/Transforms/EmulateWideInt.cpp
//===----------------------------------------------------------------------===//
FailureOr<Value> mlir::triton::getSplatValue(OpBuilder &builder,
                                              arith::ConstantOp op) {
  auto loc = op.getLoc();
  // If arith.const store a scalar type, return itself.
  if (op.getValue().getType().isIntOrFloat()) {
    return op.getResult();
  }
  Type retType = op.getType();
  auto tensorType = retType.dyn_cast_or_null<RankedTensorType>();
  if (!tensorType)
    return failure();
  auto value = op.getValue().dyn_cast<DenseElementsAttr>();
  if (!value || !value.isSplat())
    return failure();

  Type eltType = getElementTypeOrSelf(retType);
  Value fillVal =
      llvm::TypeSwitch<Type, Value>(eltType)
          .Case([&](FloatType t) {
            return builder.create<arith::ConstantOp>(
                loc, FloatAttr::get(
                         t, value.getSplatValue<APFloat>().convertToDouble()));
          })
          .Case([&](IntegerType t) {
            return builder.create<arith::ConstantOp>(
                loc, IntegerAttr::get(
                         t, value.getSplatValue<APInt>().getZExtValue()));
          })
          .Case([&](IndexType t) {
            return builder.create<arith::ConstantIndexOp>(
                loc, value.getSplatValue<APInt>().getZExtValue());
          })
          .Default([](auto) { return Value(); });
  if (!fillVal)
    return failure();
  return fillVal;
}

std::optional<Operation *> mlir::triton::getCmpSelectResult(OpBuilder &builder,
                                                     Location loc,
                                                     arith::CmpFOp op,
                                                     bool operandsSwapped) {
  auto predicate = op.getPredicate();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  switch (predicate) {
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return operandsSwapped ? builder.create<arith::MinimumFOp>(loc, lhs, rhs)
                           : builder.create<arith::MaximumFOp>(loc, lhs, rhs);
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return operandsSwapped ? builder.create<arith::MaximumFOp>(loc, lhs, rhs)
                           : builder.create<arith::MinimumFOp>(loc, lhs, rhs);
  default:
    return std::nullopt;
  }
}

std::optional<Operation *> mlir::triton::getCmpSelectResult(OpBuilder &builder,
                                                     Location loc,
                                                     arith::CmpIOp op,
                                                     bool operandsSwapped) {
  auto predicate = op.getPredicate();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  switch (predicate) {
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
    return operandsSwapped ? builder.create<arith::MinSIOp>(loc, lhs, rhs)
                           : builder.create<arith::MaxSIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    return operandsSwapped ? builder.create<arith::MinUIOp>(loc, lhs, rhs)
                           : builder.create<arith::MaxUIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
    return operandsSwapped ? builder.create<arith::MaxSIOp>(loc, lhs, rhs)
                           : builder.create<arith::MinSIOp>(loc, lhs, rhs);
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    return operandsSwapped ? builder.create<arith::MaxUIOp>(loc, lhs, rhs)
                           : builder.create<arith::MinUIOp>(loc, lhs, rhs);
  default:
    return std::nullopt;
  }
}

std::optional<Operation *> mlir::triton::getCmpSelectResult(OpBuilder &builder, Operation *cmpOp, arith::SelectOp op) {
  // Get cmp op mode.
  std::optional<arith::CmpFOp> cmpFOp;
  std::optional<arith::CmpIOp> cmpIOp;
  if (isa<arith::CmpFOp>(cmpOp)) {
    cmpFOp = cast<arith::CmpFOp>(cmpOp);
  } else if (isa<arith::CmpIOp>(cmpOp)) {
    cmpIOp = cast<arith::CmpIOp>(cmpOp);
  } else {
    return std::nullopt;
  }
  // Get specific max/min semantics.
  auto loc = op.getLoc();
  if (op->getOperand(1) == cmpOp->getOperand(0) &&
      op->getOperand(2) == cmpOp->getOperand(1)) {
    if (cmpFOp) {
      return getCmpSelectResult(builder, loc, *cmpFOp, false);
    } else if (cmpIOp) {
      return getCmpSelectResult(builder, loc, *cmpIOp, false);
    }
  } else if (op->getOperand(1) == cmpOp->getOperand(1) &&
             op->getOperand(2) == cmpOp->getOperand(0)) {
    if (cmpFOp) {
      return getCmpSelectResult(builder, loc, *cmpFOp, true);
    } else if (cmpIOp) {
      return getCmpSelectResult(builder, loc, *cmpIOp, true);
    }
  }
  return std::nullopt;
}
