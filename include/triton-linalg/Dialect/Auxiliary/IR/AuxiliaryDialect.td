//===- AuxiliaryDialect.td - Auxiliary dialect definitions--*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// Defines triton auxiliary dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_TD
#define TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_TD
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def Auxiliary_Dialect : Dialect {
  let name = "aux";
  let cppNamespace = "::mlir::triton::aux";
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
  let description = [{
    This dialect contains auxiliary operations that handle user resources
    /sign of interface/device switch.
  }];
}

#endif // TRITON_DIALECT_AUXILIARY_IR_AUXILIARYDIALECT_TD
