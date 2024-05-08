#pragma once
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"

inline void registerTritonLinalgDialects(mlir::DialectRegistry &registry) {
  // Triton.
  registry.insert<mlir::triton::TritonDialect>();
  // TritonLinalg.
  registry.insert<mlir::triton::aux::AuxiliaryDialect>();
  registry.insert<mlir::triton::linalg_ext::LinalgExtDialect>();
}
