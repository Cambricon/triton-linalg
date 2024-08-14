//===- ArithExt.h - ArithExt dialect ----------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXT_H
#define TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXT_H

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
// Arith Extension Dialect
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/ArithExt/IR/ArithExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Arith Extension Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/ArithExt/IR/ArithExtOps.h.inc"

#endif // TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXT_H
