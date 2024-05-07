//===- Pipelines.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file declares all pass pipelines
//
//===----------------------------------------------------------------------===//
#include "triton-linalg/Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "triton-linalg/Conversion/Passes.h"
#include "triton-linalg/Dialect/Arith/Transforms/Passes.h"
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace {
void buildTritonToLinalgPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::triton::createWrapFuncBodyWithSingleBlockPass());
  pm.addPass(mlir::createInlinerPass({}, nullptr));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createCanonicalizeTritonPass());
  pm.addPass(mlir::triton::createPointerStrengthReductionPass());
  // Since canonicalizer pass may convert single block function to multi-blocks,
  // we rerun this pass here.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createTritonToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::triton::createExtractLikeMoveBackwardPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createArithToLinalgPass());
  pm.addPass(mlir::triton::createMathToLinalgPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::triton::createWrapFuncBodyWithSingleBlockPass());
}
} // namespace

void ::mlir::triton::registerTritonLinalgPipelines() {
  PassPipelineRegistration<> triton_to_linalg(
      "triton-to-linalg",
      "Runs the triton to linalg dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildTritonToLinalgPipeline(passManager);
      });
}
