//===- MemRefUtils.h - Helpers related to memref Dialect --------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities and common functions for memref dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_UTILS_MEMREFUTILS_H
#define TRITON_LINALG_DIALECT_UTILS_MEMREFUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h" // IWYU pragma: keep
#include <stdint.h>

namespace mlir {
class OpBuilder;
class Operation;
class OpFoldResult;
class MLIRContext;
class ViewLikeOpInterface;
} // namespace mlir

namespace mlir {
namespace triton {

/// Try to get the broadcast dimensions from 'srcTy' to 'dstTy', if successful,
/// return the broadcast dimensions, otherwise return failure.
FailureOr<SmallVector<int64_t>> getBroadcastDimensions(MemRefType dstTy,
                                                       MemRefType srcTy);

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_UTILS_MEMREFUTILS_H
