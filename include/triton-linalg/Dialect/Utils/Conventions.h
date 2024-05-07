//===- Conventions.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_UTILS_CONVENTIONS_H
#define TRITON_LINALG_DIALECT_UTILS_CONVENTIONS_H
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <stdint.h>

namespace mlir {
class Operation;
namespace scf {
class ForallOp;
} // namespace scf
namespace triton {

/// Return the key of attr decripting whether to use tf32 to convolutional
/// computation.
constexpr llvm::StringLiteral getAttrAllowTF32() {
  return llvm::StringLiteral("__allow_tf32__");
}

/// Return the key of attr describing whether the module executes on
/// linear memory.
constexpr llvm::StringLiteral getIsLinearMemoryAttrKey() {
  return llvm::StringLiteral("triton.is_linear");
}

/// Determine whether the current module is running in linear memory space.
bool isLinearMemory(::mlir::ModuleOp op);
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_UTILS_CONVENTIONS_H
