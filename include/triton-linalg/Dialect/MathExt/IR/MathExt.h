//===- MathExt.h - MathExt dialect ------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXT_H
#define TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXT_H

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

#endif // TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXT_H
