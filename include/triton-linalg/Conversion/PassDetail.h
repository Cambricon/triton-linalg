//===- PassDetail.h - Conversion pass details file---------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_PASSDETAIL_H
#define TRITON_LINALG_CONVERSION_PASSDETAIL_H
// IWYU pragma: begin_keep
// For .inc.files.
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace func {
class FuncOp;
} // namespace func

namespace arith {
class ArithDialect;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace scf {
class SCFDialect;
} // namespace scf

namespace math {
class MathDialect;
} // namespace math

namespace shape {
class ShapeDialect;
} // namespace shape

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace triton {
class TritonDialect;
} // namespace triton

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization
// ------namespace triton------
namespace triton {
namespace aux {
class AuxiliaryDialect;
} // namespace aux

namespace linalg_ext {
class LinalgExtDialect;
} // namespace linalg_ext

// IWYU pragma: end_keep
#define GEN_PASS_CLASSES
#include "triton-linalg/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_PASSDETAIL_H
