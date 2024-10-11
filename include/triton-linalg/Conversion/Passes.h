//===- Passes.h - Conversion passes header-----------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_CONVERSION_PASSES_H
#define TRITON_LINALG_CONVERSION_PASSES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

#include "triton-linalg/Conversion/ArithToLinalg/ArithToLinalg.h"
#include "triton-linalg/Conversion/MathToLinalg/MathToLinalg.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton-linalg/Conversion/TritonToTensor/TritonToTensor.h"

namespace mlir {
class Pass;
namespace triton {
#define GEN_PASS_REGISTRATION
#include "triton-linalg/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_PASSES_H
