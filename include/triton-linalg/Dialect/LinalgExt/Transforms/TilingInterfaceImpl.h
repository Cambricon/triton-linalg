//===- TilingInterfaceImpl.h - Implementation of TilingInterface-*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_TRANSFORMS_TILINGINTERFACEIMPL_H
#define TRITON_LINALG_DIALECT_LINALGEXT_TRANSFORMS_TILINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace triton {
namespace linalg_ext {
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
void registerExtOpTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace linalg_ext
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_LINALGEXT_TRANSFORMS_TILINGINTERFACEIMPL_H
