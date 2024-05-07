//===- MathBase.td - Base definitions for math dialect -----*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef MATH_EXT_BASE
#define MATH_EXT_BASE

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
#endif // MATH_EXT_BASE
