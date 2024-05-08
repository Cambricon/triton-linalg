//===- TritonToLinalg.h - Triton to Linalg dialect convension ---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H

#include <memory>

namespace mlir {
class Pass;
namespace triton {

/// Create a pass to convert a subset of Triton ops to Linalg.
std::unique_ptr<mlir::Pass> createTritonToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
