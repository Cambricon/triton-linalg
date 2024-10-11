//===- MathExtBase.td - Base definitions of MathExt dialect *- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXTBASE_TD
#define TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXTBASE_TD

include "mlir/IR/OpBase.td"

def MathExt_Dialect : Dialect {
  let name = "math_ext";
  let cppNamespace = "::mlir::math_ext";
  let description = [{
    Note:
       This dialect is considered as complements for public math dialect.
  }];
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}
#endif // TRITON_LINALG_DIALECT_MATHEXT_IR_MATHEXTBASE_TD
