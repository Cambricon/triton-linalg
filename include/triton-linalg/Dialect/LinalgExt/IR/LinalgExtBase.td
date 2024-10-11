//===- LinalgExtBase.td - Linalg ext base statements -------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTBASE_TD
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTBASE_TD

include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtEnums.td"
include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtInterface.td"
include "triton-linalg/Interfaces/InferResultTypeOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"

def LinalgExt_Dialect : Dialect {
  let name = "linalg_ext";
  let description = [{
    The `linalg_ext` dialect is intended to experiment more support for
    structured operations, ie, can not be represented in Linalg operations or
    some operations to be added in the next versions.
  }];
  let cppNamespace = "::mlir::triton::linalg_ext";
  let dependentDialects = [
    "arith::ArithDialect",
    "affine::AffineDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
    "linalg::LinalgDialect"
  ];
  let hasOperationAttrVerify = 1;
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    using RegionBuilderFunType = llvm::function_ref<
      void(ImplicitLocOpBuilder &b, Block &, ArrayRef<NamedAttribute>)>;
    RegionBuilderFunType getRegionBuilder(StringRef name) {
      return namedStructuredOpRegionBuilders.lookup(name);
    }
    private:
      llvm::StringMap<RegionBuilderFunType> namedStructuredOpRegionBuilders;
  }];
}

class LinalgExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// LinalgExt base class.
//===----------------------------------------------------------------------===//
class LinalgExtBase_Op<string mnemonic, list<Trait> props>
  : Op<LinalgExt_Dialect, mnemonic, !listconcat([
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgExtInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  let hasFolder = 1;
  let hasVerifier = 1;
  code extraLinalgExtOpClassDeclaration = [{}];
}

//===----------------------------------------------------------------------===//
// Structured Base class.
//===----------------------------------------------------------------------===//
// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on ShapedType as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgStructuredBase_Op<string mnemonic, list<Trait> props>
  : Op<LinalgExt_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indice.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
        return llvm::cast<linalg::LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  }];
  let hasFolder = 1;
}

def TensorOrMemref :
    AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;
#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTBASE_TD
