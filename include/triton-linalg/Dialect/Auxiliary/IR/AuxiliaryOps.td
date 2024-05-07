//===- AuxiliaryOps.td - Auxiliary operation definitions ---*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// Defines triton auxiliary operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_AUXILIARY_IR_AUXILIARYOPS_TD
#define TRITON_DIALECT_AUXILIARY_IR_AUXILIARYOPS_TD

include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.td"
include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

// Base class for Auxiliary dialect ops.
class Aux_Op<string mnemonic, list<Trait> traits = []> :
    Op<Auxiliary_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// StoreResourceOp
//===----------------------------------------------------------------------===//
def StoreResourceOp : Aux_Op<"store"> {
  let summary =
      "Stores a value to a tensor gained from virtual resource handle.";
  let description = [{
    An operation with side-effect that represents a store operation on tensor
    level. 'to' must comes from a loadresource op/tensor view op.
    A 'write' attr is aquired from loadresource.

    Example:
    ```
    %0 = aux.load(0) : tensor<1xf32>
    %c1 = arith.constant 1 : tensor<1xf32>
    aux.store %c1, %0: tensor<1xf32> to tensor<1xf32>
    ```
  }];

  let arguments = (ins
    AnyType:$to,
    AnyType:$from
  );

  let results = (outs);

  let assemblyFormat = [{
    $from `,` $to `:`
    type($from) `to` type($to)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isScalar(const Value& value) {
      return !value.getType().isa<ShapedType>();
    }

    bool hasPureBufferSemantics() {
      return ::llvm::all_of(getOperands(),
        [&](const Value& opOperand) {
          return isScalar(opOperand) ||
                 opOperand.getType().isa<::mlir::MemRefType>();
        });
    }

    bool hasPureTensorSemantics() {
      return ::llvm::all_of(getOperands(),
        [&](const Value& opOperand) {
          return isScalar(opOperand) ||
                 opOperand.getType().isa<::mlir::TensorType>();
        });
    }

  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//
def ViewOp :
  Aux_Op<"view", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      AttrSizedOperandSegments,
      OffsetSizeAndStrideOpInterface,
      ViewLikeOpInterface]> {
  let summary = "To view a ptr as a memref.";
  let description = [{
    Attach offset, sizes, strides to a llvm pointer.
    Example:
    ```mlir
    aux.view %ptr to
      offset: [0],
      sizes: [%size0, 10],
      strides: [1, %stride1]
    : llvm.ptr<f32> to memref<?x10xf32, strided<[1, ?], offset: 0>>
    ```

    This operation creates a new memref descriptor using the base of the
    ptr and applying the input arguments to the other metadata.
    In other words:
    ```mlir
    %dst = aux.view %src to
      offset: [%offset],
      sizes: [%sizes],
      strides: [%strides]
    ```
    means that `%dst`'s descriptor will be:
    ```mlir
    %dst.base = %src
    %dst.aligned = %src
    %dst.offset = %offset
    %dst.sizes = %sizes
    %dst.strides = %strides
    ```
  }];

  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    OptionalAttr<StrAttr>:$cache_mode);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $ptr `to` `offset` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `sizes` `` `:`
    custom<DynamicIndexList>($sizes, $static_sizes)
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($ptr) `to` type($result)
  }];

  let builders = [
    // Build a ViewOp with mixed static and dynamic entries.
    OpBuilder<(ins "MemRefType":$resultType, "Type":$elementType,
      "Value":$source, "OpFoldResult":$offset, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "StringAttr": $cacheMode,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ViewOp with static entries.
    OpBuilder<(ins "MemRefType":$resultType, "Type":$elementType,
      "Value":$source, "int64_t":$offset, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides,
      "StringAttr": $cacheMode,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ViewOp with dynamic entries.
    OpBuilder<(ins "MemRefType":$resultType, "Type":$elementType,
      "Value":$source, "Value":$offset, "ValueRange":$sizes,
      "ValueRange":$strides,
      "StringAttr": $cacheMode,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ViewOp with dynamic entries and inferred result type.
    OpBuilder<(ins "Type":$elementType, "Value":$source, "Value":$offset,
      "ValueRange":$sizes, "ValueRange":$strides,
      "StringAttr": $cacheMode,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "Type":$elementType, "Value":$source, "OpFoldResult":$offset,
      "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "StringAttr": $cacheMode,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }

    // The result of the op is always a ranked memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    Value getViewSource() { return getPtr(); }
    Value getOffset() {
      return getOffsets().empty() ? nullptr : getOffsets()[0];
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = getResult().getType().cast<ShapedType>().getRank();
      return {1, resultRank, resultRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

def OptimizationBarrierOp : Aux_Op<"optimization_barrier", []> {
  let summary = "Prevents compiler optimizations across a value.";
  let description = [{
    Wraps any operands in an unoptimizable identity to prevent optimized.
    It will be dropped during the final step in compilation and
    has no effect at runtime.
  }];
  let arguments = (ins Arg<AnyType, "do not optimize", [MemWrite]>:$operand);

  let assemblyFormat = [{
    attr-dict
    $operand `:` type($operand)
  }];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Aux_Op<"print", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                               DestinationStyleOpInterface]> {
  let summary = "print operator";
  let description = [{
    `aux.print` takes a literal string prefix and a tensor or memref argument that should be printed

    - `format`: *Optional*, formatted string. If provided, it prints the elements in the specified user-defined format; otherwise, the format is generated automatically from the arguments. Only one valid format specifier is accepted.

    Example:
    ```
    %0 = aux.print(%data : tensor<128xi8>) -> (tensor<128xi8>)
    %1 = aux.print(%data : tensor<128xi8>) {format = "%d"} -> (tensor<128xi8>)
    
    aux.print(%src: memref<128x16xf16, 101>) {format = "src: %hf\n"}
    aux.print(%src: memref<128x16xf32>)

    ```
  }];
  let arguments = (ins Variadic<AnyType>:$values,
                       OptionalAttr<StrAttr>:$format);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    (`(` $values^ `:` type($values) `)` )?
    attr-dict
    (`->` `(` type($result)^ `)`)?
  }];

  let extraClassDeclaration = [{
    bool hasPureBufferSemantics() {
      return ::llvm::all_of(getOperands(),
        [&](const Value& opOperand) {
          return opOperand.getType().isa<::mlir::MemRefType>();
        });
    }
    
    ShapedType getInitType() {
      return getOperands()[0].getType().cast<ShapedType>();;
    }

    MutableOperandRange getDpsInitsMutable() { return getValuesMutable(); }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScalarPrintOp
//===----------------------------------------------------------------------===//

def ScalarPrintOp : Aux_Op<"scalar.print", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "print operator";
  let description = [{
    `aux.scalar.print` takes a literal string prefix and one scalar argument that should be printed

    - `format`: *Optional*, formatted string. If provided, it prints the elements in the specified user-defined format; otherwise, the format is generated automatically form the ``%values``.


    Example:
    ```
    aux.scalar.print {format = "print test"}
    
    aux.scalar.print(%data : i32)
    aux.scalar.print(%data : i32) {format = "data: %d\n"}
    aux.scalar.print(%data : i32) {format = "data: "}
    ```
  }];
  let arguments = (ins Optional<AnyType>:$values,
                       OptionalAttr<StrAttr>:$format);
  let assemblyFormat = [{
    (`(` $values^ `:` type($values) `)` )?
    attr-dict
  }];
  let hasVerifier = 1;
}
#endif // TRITON_DIALECT_AUXILIARY_IR_AUXILIARYOPS_TD
