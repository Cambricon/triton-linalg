//===- MathOps.td - Math op definitions --------------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef MATH_OPS
#define MATH_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "triton-linalg/Dialect/MathExt/IR/MathBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for math extension dialect ops. 
class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;

// Base class for binary math operations on integer types. Require two
// operands and one result of the same type. This type can be an integer
// type, vector or tensor thereof.
class MathExt_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> :
    MathExt_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MulhiUIOp
//===----------------------------------------------------------------------===//

def MathExt_MulhiUIOp : MathExt_IntegerBinaryOp<"mulhiui"> {
  let summary = "signed integer raised to the power of operation";
  let description = [{
    Most significant N bits of the 2N-bit product of two integers.
  }];
}

#endif // MATH_OPS
