//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSDETAIL_H
#define TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSDETAIL_H
// For .inc.files.
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"

namespace mlir {
class DialectRegistry;
class ModuleOp;
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace scf {
class SCFDialect;
} // namespace scf

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace triton {
class TritonDialect;
} // namespace triton

namespace func {
class FuncDialect;
class FuncOp;
} // namespace func

namespace triton {
namespace linalg_ext {
class LinalgExtDialect;
} // namespace linalg_ext

namespace aux {
class AuxiliaryDialect;
} // namespace aux
} // namespace triton

namespace triton {
#define GEN_PASS_CLASSES
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSDETAIL_H
