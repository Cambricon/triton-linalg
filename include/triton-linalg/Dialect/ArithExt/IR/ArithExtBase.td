//===- ArithBase.td - Base definitions for ArithExt dialect *- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD
#define TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD

include "mlir/IR/OpBase.td"

def ArithExt_Dialect : Dialect {
  let name = "arith_ext";
  let cppNamespace = "::mlir::triton::arith_ext";
  let description = [{
    Note:
       This dialect is considered as complements for public arith dialect.
  }];
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}
#endif // TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD
