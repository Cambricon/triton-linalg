#pragma once
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/Auxiliary/Transforms/AuxOpTilingInterface.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/LinalgExt/Transforms/TilingInterfaceImpl.h"
#include "triton-linalg/Transforms/InferAxisInfoInterfaceImpl.h"


inline void registerTritonLinalgDialects(mlir::DialectRegistry &registry) {
  // Triton.
  registry.insert<mlir::triton::TritonDialect>();
  // TritonLinalg.
  registry.insert<mlir::triton::aux::AuxiliaryDialect>();
  registry.insert<mlir::triton::linalg_ext::LinalgExtDialect>();

  mlir::triton::aux::registerTilingInterfaceExternalModels(registry);
  mlir::triton::linalg_ext::registerTilingInterfaceExternalModels(registry);
  mlir::triton::linalg_ext::registerExtOpTilingInterfaceExternalModels(registry);
  mlir::triton::registerInferAxisInfoInterfaceExternalModels(registry);
}
