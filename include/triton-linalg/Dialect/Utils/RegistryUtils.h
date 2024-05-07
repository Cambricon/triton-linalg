//===- RegistryUtils.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_UTILS_REGISTRYUTILS_H
#define TRITON_LINALG_DIALECT_UTILS_REGISTRYUTILS_H

#include <initializer_list>
namespace mlir {
class MLIRContext;
namespace triton {
namespace utils {

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `interface` with each of them.
template <template <typename> typename Interface, typename... Ops>
struct AttachInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (void)std::initializer_list<int>{
        0, (Ops::template attachInterface<Interface<Ops>>(*ctx), 0)...};
  }
};

} // namespace utils
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_UTILS_REGISTRYUTILS_H
