//===- InferAxisInfoInterfaceImpl.h -----------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_INFERAXISINFOINTERFACEIMPL_H
#define TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_INFERAXISINFOINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace triton {

void registerInferAxisInfoInterfaceExternalModels(DialectRegistry &registry);

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_INFERAXISINFOINTERFACEIMPL_H
