//===- MathOps.cpp - MLIR operations for math implementation --------------===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "include/triton-linalg/Dialect/MathExt/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include <optional>

using namespace mlir;
using namespace mlir::math_ext;

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "include/triton-linalg/Dialect/MathExt/IR/MathExtOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MulhiUIOp
//===----------------------------------------------------------------------===//

/// Materialize an integer or floating point constant.
Operation *math_ext::MathExtDialect::materializeConstant(OpBuilder &builder,
                                                         Attribute value,
                                                         Type type,
                                                         Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return builder.create<ub::PoisonOp>(loc, type, poison);

  return arith::ConstantOp::materialize(builder, value, type, loc);
}
