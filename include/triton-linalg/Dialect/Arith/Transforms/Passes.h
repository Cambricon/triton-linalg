//===- Passes.h - Passes for arith ------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_H
#define TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton-linalg/Dialect/Arith/Transforms/PassDetail.h"

namespace mlir {
namespace triton {
namespace arith_ext {

std::unique_ptr<Pass> createArithCanonicalizerPass();

#define GEN_PASS_REGISTRATION
#include "triton-linalg/Dialect/Arith/Transforms/Passes.h.inc"

} // namespace arith_ext
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_H
