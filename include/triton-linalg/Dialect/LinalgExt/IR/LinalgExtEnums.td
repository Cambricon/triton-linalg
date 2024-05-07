//===- LinalgExtEnums.td -Linalg ext enums------------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This is the additional attribute definition file.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTENUMS_TD
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTENUMS_TD

include "mlir/IR/EnumAttr.td"
// The compute accuracy to control what data type to use to calculate convolution.
def LinalgExt_AllowTF32: I32EnumAttr<
    "AllowTF32", "",
    [
      I32EnumAttrCase<"not_allow_tf32", 0>,
      I32EnumAttrCase<"allow_tf32", 1>,
    ]> {
  let cppNamespace = "::mlir::triton::linalg_ext";
}

def LinalgExt_AtomicTypeAttr: I32EnumAttr<
    "AtomicType", "",
    [
      I32EnumAttrCase<"addf", 1>,
      I32EnumAttrCase<"addi", 2>,
      I32EnumAttrCase<"andi", 3>,
      I32EnumAttrCase<"ori", 4>,
      I32EnumAttrCase<"xori", 5>,
      I32EnumAttrCase<"maximumf", 6>,
      I32EnumAttrCase<"maxs", 7>,
      I32EnumAttrCase<"maxu", 8>,
      I32EnumAttrCase<"minimumf", 9>,
      I32EnumAttrCase<"mins", 10>,
      I32EnumAttrCase<"minu", 11>,
      I32EnumAttrCase<"xchg", 12>,
    ]> {
  let cppNamespace = "::mlir::triton::linalg_ext";
}
#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTENUMS_TD
