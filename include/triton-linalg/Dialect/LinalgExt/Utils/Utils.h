//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H
#define TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H

namespace mlir {
class Block;
class Operation;
namespace triton {
namespace linalg_ext {
/// Retrieve the operation from the body, if it is the only one (except
/// yield) and if it gets the same amount of arguments as the body does.
/// If initFirst flag is enabled, we check that init takes the first position in
/// operands of payload.
Operation *findPayloadOp(Block *body, bool initFirst = false);
} // namespace linalg_ext
} // namespace triton
} // namespace mlir
#endif // TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H
