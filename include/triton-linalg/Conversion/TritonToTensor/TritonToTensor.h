//===- TritonToTensor.h - Triton to Tensor dialect convension ---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOTENSOR_TRITONTOTENSOR_H
#define TRITON_LINALG_CONVERSION_TRITONTOTENSOR_TRITONTOTENSOR_H

#include <memory>

namespace mlir {
class Pass;
namespace triton {
/// Create a pass to convert a subset of Triton ops to Tensor ops.
std::unique_ptr<mlir::Pass> createTritonToTensorPass();
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOTENSOR_TRITONTOTENSOR_H
