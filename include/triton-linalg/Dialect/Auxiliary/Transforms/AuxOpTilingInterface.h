//===- AuxOpTilingInterface.h -Impl of AuxOpTilingInterface------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_AUXILIARY_TRANSFORMS_AUXOPTILINGINTERFACE_H
#define TRITON_LINALG_DIALECT_AUXILIARY_TRANSFORMS_AUXOPTILINGINTERFACE_H

namespace mlir {
class DialectRegistry;

namespace triton {
namespace aux {
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace aux
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_AUXILIARY_TRANSFORMS_AUXOPTILINGINTERFACE_H
