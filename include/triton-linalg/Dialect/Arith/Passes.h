//===- Passes.h - Passes for arith ------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITH_PASSES_H
#define TRITON_LINALG_DIALECT_ARITH_PASSES_H

#include "triton-linalg/Dialect/Arith/PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace triton {
namespace arith_ext {

std::unique_ptr<Pass> createArithCanonicalizerPass();

#define GEN_PASS_REGISTRATION
#include "triton-linalg/Dialect/Arith/Passes.h.inc"

} // namespace arith_ext
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_ARITH_PASSES_H
