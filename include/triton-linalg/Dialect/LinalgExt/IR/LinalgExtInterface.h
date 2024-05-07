//===- LinalgExtInterface.h -[Desc]------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_H
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_H

// IWYU pragma: begin_keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
// IWYU pragma: end_keep

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir {
namespace triton {

namespace detail {

LogicalResult verifyLinalgExtOpInterface(Operation *op);

} // namespace detail
} // namespace triton
} // namespace mlir

/// Include the generated interface declarations.
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtInterface.h.inc" // IWYU pragma: export

#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACE_H
