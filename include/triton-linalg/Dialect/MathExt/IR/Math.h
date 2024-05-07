//===- Math.h - Math dialect ------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_EXT_IR_MATH_H_
#define MLIR_DIALECT_MATH_EXT_IR_MATH_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Math Extension Dialect
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/MathExt/IR/MathExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Extension Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/MathExt/IR/MathExtOps.h.inc"

#endif // MLIR_DIALECT_MATH_EXT_IR_MATH_H_
