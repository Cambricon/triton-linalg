//===- Passes.td - Passes for arith ------------------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_TD
#define TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ArithCanonicalizer: Pass<"arith-canonicalize"> {
  let summary = "Register extra canonicalize patterns for arith ops.";
  let constructor = "mlir::triton::arith_ext::createArithCanonicalizerPass()";
  let dependentDialects = ["arith::ArithDialect"];
}

#endif // TRITON_LINALG_DIALECT_ARITH_TRANSFORMS_PASSES_TD
