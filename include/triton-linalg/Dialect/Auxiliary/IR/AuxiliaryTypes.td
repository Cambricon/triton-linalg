//===- AuxiliaryTypes.td - Auxiliary dialect type ----------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// Defines triton auxiliary dialect type.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_DIALECT_AUXILIARY_IR_AUXILIARYTYPES_TD
#define TRITON_DIALECT_AUXILIARY_IR_AUXILIARYTYPES_TD
include "mlir/IR/EnumAttr.td"

def ResourceUsageAttr : I64EnumAttr<
    "ResourceUsage", "",
    [
      I64EnumAttrCase<"R", 0, "read">,
      I64EnumAttrCase<"W", 1, "write">,
      I64EnumAttrCase<"RW", 2, "read_write">,
    ]> {
  let cppNamespace = "::mlir::triton::aux";
}

#endif // TRITON_DIALECT_AUXILIARY_IR_AUXILIARYTYPES_TD
