//===- Passes.td - Conversion passes def file---------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_CONVERSION_PASSES_TD
#define TRITON_LINALG_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ArithToLinalgPass : Pass<"convert-arith-to-linalg"> {
  let summary = "Convert the operations from the Arith dialect into the Linalg dialect";
  let constructor = "mlir::triton::createArithToLinalgPass()";
  let dependentDialects = [
      "linalg::LinalgDialect", "arith::ArithDialect", "linalg_ext::LinalgExtDialect"
  ];
}

def MathToLinalgPass : Pass<"convert-math-to-linalg"> {
  let summary = "Convert the operations from the Math dialect into the Linalg dialect";
  let constructor = "mlir::triton::createMathToLinalgPass()";
  let dependentDialects = [
      "linalg::LinalgDialect", "linalg_ext::LinalgExtDialect", "math::MathDialect",
  ];
}

def TritonToTensorPass : Pass<"convert-triton-to-tensor", "ModuleOp"> {
  let summary = "Convert the operations from the Triton dialect into the Tensor dialect";
  let constructor = "mlir::triton::createTritonToTensorPass()";
  let dependentDialects = [
      "triton::TritonDialect", "tensor::TensorDialect",
  ];
}

def TritonToLinalgPass : Pass<"convert-triton-to-linalg", "ModuleOp"> {
  let summary = "Convert the operations from the Triton dialect into the Linalg dialect";
  let constructor = "mlir::triton::createTritonToLinalgPass()";
  let dependentDialects = [
      "triton::TritonDialect", "linalg::LinalgDialect",
      "linalg_ext::LinalgExtDialect", "scf::SCFDialect",
      "arith::ArithDialect", "triton::aux::AuxiliaryDialect",
      "bufferization::BufferizationDialect",
  ];
}
#endif // TRITON_LINALG_CONVERSION_PASSES_TD
