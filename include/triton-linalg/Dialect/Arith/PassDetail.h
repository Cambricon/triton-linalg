//===- PassDetail.h - Details for arith transforms --------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITH_PASSDETAIL_H
#define TRITON_LINALG_DIALECT_ARITH_PASSDETAIL_H
// IWYU pragma: begin_keep
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace triton {
namespace arith_ext {
// IWYU pragma: end_keep
#define GEN_PASS_CLASSES
#include "triton-linalg/Dialect/Arith/Passes.h.inc"

} // namespace arith_ext
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_ARITH_PASSDETAIL_H
