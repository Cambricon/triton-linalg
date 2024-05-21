//===- AuxiliaryDialect.h - MLIR Dialect for auxiliary ops ------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
// This file defines the auxiliary operations.
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_H
#define TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_H
// IWYU pragma: begin_keep
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// IWYU pragma: end_keep
//===----------------------------------------------------------------------===//
// AuxiliaryDialect
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Auxiliary Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Auxiliary Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOps.h.inc"

#endif // TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_H
