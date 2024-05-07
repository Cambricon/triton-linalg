//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// triton transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_H
#define TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h" // IWYU pragma: keep
#include <memory>

namespace mlir {
class RewritePatternSet;
namespace triton {

// Create a pass to canonicalize triton ir.
std::unique_ptr<Pass> createCanonicalizeTritonPass();

/// Create a pass to move backward extract-like operations.
std::unique_ptr<Pass> createExtractLikeMoveBackwardPass();

/// Create a pass to deal with triton operations with ptr.
std::unique_ptr<Pass> createPointerStrengthReductionPass();

/// Create a pass to wrap function body with a block.
std::unique_ptr<Pass> createWrapFuncBodyWithSingleBlockPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

// Include the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_H
