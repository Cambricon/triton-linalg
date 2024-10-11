//===- LinalgExtOps.h - LinalgExt dialect -----------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H

// IWYU pragma: begin_keep
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtInterface.h"
// IWYU pragma: end_keep
//===----------------------------------------------------------------------===//
// LinalgExt Dialect
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// LinalgExt Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtEnums.h.inc"

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h.inc"

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtStructedOps.h.inc"
#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_H
