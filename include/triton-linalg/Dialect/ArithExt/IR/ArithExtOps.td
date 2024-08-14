//===- ArithExtOps.td - ArithExt op definitions ------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD
#define TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "triton-linalg/Dialect/ArithExt/IR/ArithExtBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for arith extension dialect ops. 
class ArithExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<ArithExt_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for binary arith operations on floating types. Require two
// operands and one result of the same type. This type can be an floating
// type, vector or tensor thereof.
class ArithExt_FloatingBinaryOp<string mnemonic, list<Trait> traits = []> :
    ArithExt_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs FloatLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MinFirstFOp
//===----------------------------------------------------------------------===//

def ArithExt_MinFirstFOp : ArithExt_FloatingBinaryOp<"minfirstf"> {
  let summary = "min operation, which supports NaN.";
  let description = [{
    Returns the minimum of the two arguments, −0 and +0 compare as equal.
    If one of the arguments is NaN, then the result is the first operand.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = arith_ext.minfirstf %b, %c : f32
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxFirstFOp
//===----------------------------------------------------------------------===//

def ArithExt_MaxFirstFOp : ArithExt_FloatingBinaryOp<"maxfirstf"> {
  let summary = "max operation, which supports nan.";
  let description = [{
    Returns the maximum of the two arguments, −0 and +0 compare as equal.
    If one of the arguments is NaN, then the result is the first operand.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = arith_ext.maxfirstf %b, %c : f32
    ```
  }];
}
#endif // TRITON_LINALG_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD
