//===- LinalgExtOps.td - Linalg ext ops --------------------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This is the additional operation definition file for structured operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_TD
#define TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_TD

include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
//===----------------------------------------------------------------------===//
// Op definition for ScatterOp
//===----------------------------------------------------------------------===//
def ScatterOp : LinalgExtBase_Op<"scatter", []> {

  let summary = [{ Scatter computation operation. }];
  let description = [{
    Scatter has two to three inputs: update, indice and mask.

    Init has shape [i0, i1, ..., in-1], indice has shape [Batch0, Batch1, ..., Batchm-1, k], update
    has shape [Batch0, Batch1, ..., Batchm-1, window0, ..., windown-1], mask has shape [Batch0, Batch1, ..., Batchm-1].
    k should less equal to n.
    A complete index should be computed by dimension_map and indice. For each
    batch of indice we have [idx0, idx1... idxk] with same length of dimension_map,
    index computation is as follow:
    ```
    SmallVector<int64_t> wholeIdx(n);
    for (int i = 0; i < dimension_map.size(); ++i) {
      wholeIdx[dimension_map[i]] = indice[i];
    }
    ```

    The whole scatter computation is:
    ```
    for (i0 = 0; i0 < Batch0; ++i0) {
      ...
      for (im-1 = 0; im-1 < Batchm-1; ++im-1) {
        indice = wholeIdx[i0, ..., im-1];
        if (mask[i0, ..., im-1]) {
          // each computation will eventually effect window size of init.
          computation(init[indice], update[i0, ..., im-1]);
          // if region is empty, only copy will apply on init.
        }
      }
    }
    ```
    The dimension_map attributes describes which index value maps to which
    dimension in the destionation. It cannot contain duplicate values, must
    have as many entries as index depth, and values must be within the rank of
    the destination.

    The ranged_data attribute carries the information whether input data's size
    can be reached during runtime. If there is limited input data, it can be tile
    with iteration loops marked as parallel.

    The overlap_window attribute carries the information whether each batch of window
    's computation will overlap. If there are overlap computation, the first iteration
    loop will be marked as reduction and can not be tiled.
  }];

    let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      TensorOrMemref:$init,
      DenseI64ArrayAttr:$dimension_map,
      DefaultValuedAttr<BoolAttr, "true">:$ranged_data,
      DefaultValuedAttr<BoolAttr, "true">:$overlap_window,
      DefaultValuedAttr<BoolAttr, "true">:$signed_indice
    );
    let results = (outs Variadic<AnyTensor>:$result);
    let regions = (region SizedRegion<1>:$region);

    let assemblyFormat = [{
      attr-dict `dimension_map` `=` $dimension_map
                `ranged_data` `(` $ranged_data `)`
                `overlap_window` `(` $overlap_window `)`
                `signed_indice` `(` $signed_indice `)`
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `outs` `(` $init `:` type($init) `)`
      $region (`->` type($result)^)?
    }];
    let hasFolder = 1;
    let builders = [
      OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
        "ArrayRef<int64_t>": $dimensionMap, "bool":$rangedData,
        "bool":$overlapWindow,
        "bool":$signed_indice,
        CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "{}">:$region,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
    ];
    let extraClassDeclaration = [{
      // base helper function
      Value update() {
        return getDpsInputOperand(0)->get();
      }
      Value indice() {
        return getDpsInputOperand(1)->get();
      }
      Value mask() {
        if (getInputs().size() > 2) {
          return getDpsInputOperand(2)->get();
        }
        return Value();
      }
      ShapedType getUpdateType() {
        return update().getType().cast<ShapedType>();
      }
      ShapedType getIndiceType() {
        return indice().getType().cast<ShapedType>();
      }
      ShapedType getInitType() {
        return getInit().getType().cast<ShapedType>();
      }
      ShapedType getMaskType() {
        if (mask()) {
          return mask().getType().cast<ShapedType>();
        }
        return {};
      }
      int64_t getIndexDepth() {
        return getIndiceType()
            .getShape()
            .back();
      }
      int64_t getBatchDimNum() {
        return getIndiceType().getRank() - 1;
      }
      LogicalResult reifyResultShapes(OpBuilder &b,
                                      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
        return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
      }
      MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for GatherOp
//===----------------------------------------------------------------------===//
def GatherOp : LinalgExtBase_Op<"gather", []> {

  let summary = [{ Gather computation operation. }];
  let description = [{
    Gather has two or three inputs: input, indice, mask.
    Input has shape [i0, i1, ..., in-1], indice has shape [Batch0, Batch1, ..., Batchm-1, k],
    init has shape [Batch0, Batch1, ..., Batchm-1, o0, o1, ..., on-1].
    k should less equal to n.
    A complete index should be computed by dimension_map and indice. For each
    batch of indice we have [idx0, idx1... idxk] with same length of dimension_map,
    index computation is as follow:
    ```
    SmallVector<int64_t> wholeIdx(n);
    for (int i = 0; i < dimension_map.size(); ++i) {
      wholeIdx[dimension_map[i]] = indice[i];
    }
    ```
    The whole gather computation is:
    ```
    for (i0 = 0; i0 < Batch0; ++i0) {
      ...
      for (im-1 = 0; im-1 < Batchm-1; ++im-1) {
        indice = indice[i0, ..., im-1];
        if (mask[i0, ..., im-1]) {
          // each computation will eventually effect window size of init.
          computation(input[indice], init[i0, ..., im-1]);
          // if region is empty, only copy will apply on init.
        }
      }
    }
    ```
    The dimension_map attributes describes which index value maps to which
    dimension in the destionation. It cannot contain duplicate values, must
    have as many entries as index depth, and values must be within the rank of
    the destination.

    The ranged_data attribute carries the information whether input data's size
    can be reached during runtime. If there is limited input data, it can be tile
    with iteration loops marked as parallel.
  }];

    let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      TensorOrMemref:$init,
      DenseI64ArrayAttr:$dimension_map,
      DefaultValuedAttr<BoolAttr, "true">:$ranged_data,
      DefaultValuedAttr<BoolAttr, "true">:$signed_indice
    );
    let results = (outs Variadic<AnyTensor>:$result);
    let regions = (region SizedRegion<1>:$region);

    let assemblyFormat = [{
      attr-dict `dimension_map` `=` $dimension_map
                `ranged_data` `(` $ranged_data `)`
                `signed_indice` `(` $signed_indice `)`
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `outs` `(` $init `:` type($init) `)`
      $region (`->` type($result)^)?
    }];
    let hasFolder = 1;
    let builders = [
      OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
        "ArrayRef<int64_t>": $dimensionMap,
        "bool":$rangedData, "bool":$signed_indice,
        CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "{}">:$region,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
    ];
    let extraClassDeclaration = [{
      // base helper function
      Value input() {
        return getDpsInputOperand(0)->get();
      }
      Value indice() {
        return getDpsInputOperand(1)->get();
      }
      Value mask() {
        if (getInputs().size() > 2) {
          return getDpsInputOperand(2)->get();
        }
        return Value();
      }
      ShapedType getInputType() {
        return input().getType().cast<ShapedType>();
      }
      ShapedType getIndiceType() {
        return indice().getType().cast<ShapedType>();
      }
      ShapedType getMaskType() {
        if (mask()) {
          return mask().getType().cast<ShapedType>();
        }
        return {};
      }
      int64_t getIndexDepth() {
        return getIndiceType()
            .getShape()
            .back();
      }
      ShapedType getInitType() {
        return getInit().getType().cast<ShapedType>();
      }
      int64_t getBatchDimNum() {
        return getIndiceType().getRank() - 1;
      }
      LogicalResult reifyResultShapes(OpBuilder &b,
                                      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
        return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
      }
      MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for contiguous AtomicRMWOp
//===----------------------------------------------------------------------===//
def AtomicRMWOp : LinalgExtBase_Op<"atomic_rmw", [AttrSizedOperandSegments, SameOperandsElementType, SameOperandsShape]> {
  let summary = [{ AtomicRMW computation operation. }];
  let description = [{
    AtomicRMW has one input, which is the rhs to perform the modification.
    Input has shape [i0, i1, ..., in-1].
    It has two inits: src is the tensor to be rmw with the same shape as input,
                      dst is the returned original input with the same shape as input.
    The atomic_type indicates the specific atomic computation to be executed.

    The whole AtomicRMW computation is:
    ```
    // Read from src, add input to src, and
    // store original value of src to dst.
    // linalg_ext.atomic_rmw addf ins(%alloc_5 : memref<4x1xf32, 101>) outs(%view_memref, %alloc_4 : memref<4xf32, 1>, memref<4x1xf32, 101>) -> memref<4xf32, 1>, memref<4x1xf32, 101>

    ```
  }];

    let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$inits,
      LinalgExt_AtomicTypeAttr:$atomic_type,
      LinalgExt_MemoryOrder:$memory_order
    );
    let results = (outs Variadic<TensorOrMemref>:$result);

    let assemblyFormat = [{
      $atomic_type $memory_order attr-dict
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `outs` `(` $inits `:` type($inits) `)`
       (`->` type($result)^)?
    }];
    let hasFolder = 1;
    let builders = [
      OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
        "AtomicType": $atomic_type, "MemoryOrder": $memory_order,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
    ];
    let extraClassDeclaration =
      extraLinalgExtOpClassDeclaration # [{
      // Base helper function.
      Value input() {
        return getDpsInputOperand(0)->get();
      }
      Value src() {
        return getDpsInitOperand(0)->get();
      }
      Value dst() {
        return getDpsInitOperand(1)->get();
      }
      ShapedType getInputType() {
        return input().getType().cast<ShapedType>();
      }
      ShapedType getSrcType() {
        return src().getType().cast<ShapedType>();
      }
      LogicalResult reifyResultShapes(OpBuilder &b,
                                      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
        return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
      }
      MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for GatherAtomicRMWOp
//===----------------------------------------------------------------------===//
def GatherAtomicRMWOp : LinalgExtBase_Op<"gather_atomic_rmw", [AttrSizedOperandSegments]> {
  let summary = [{ GatherAtomicRMW computation operation. }];
  let description = [{
    GatherAtomicRMW has two or three inputs: input, indice, mask.
    Input is the rhs to perform the modification.
    Input has shape [Batch, i0, i1, ..., in-1], indice has shape [Batch, n],
    mask if exists has shape [Batch].
    It has two inits: src is the tensor to be rmw rank is n,
                      window is the returned original input with shape [Batch, i0, i1, ..., in-1].
    The atomic_type indicates the specific atomic computation to be executed.

    The whole GatherAtomicRMW computation is:
    ```
    for (count = 0; count < Batch; ++count) {
      indice = indice[count];
      if (mask[count]) {
        // Read from src, add input[count] to src, and
        // store original value of src to window[count].
        atomicRmw(window[count], src[count], input[count]);
      }
    }
    ```
  }];

    let arguments = (ins
      Variadic<TensorOrMemref>:$inputs,
      Variadic<TensorOrMemref>:$inits,
      LinalgExt_AtomicTypeAttr:$atomic_type,
      LinalgExt_MemoryOrder:$memory_order
    );
    let results = (outs Variadic<TensorOrMemref>:$result);

    let assemblyFormat = [{
      $atomic_type $memory_order attr-dict
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `outs` `(` $inits `:` type($inits) `)`
       (`->` type($result)^)?
    }];
    let hasFolder = 1;
    let builders = [
      OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
        "AtomicType": $atomic_type, "MemoryOrder": $memory_order,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
    ];
    let extraClassDeclaration =
      extraLinalgExtOpClassDeclaration # [{
      // Base helper function.
      Value input() {
        return getDpsInputOperand(0)->get();
      }
      Value indice() {
        return getDpsInputOperand(1)->get();
      }
      Value mask() {
        if (getInputs().size() > 2) {
          return getDpsInputOperand(2)->get();
        }
        return Value();
      }
      Value src() {
        return getDpsInitOperand(0)->get();
      }
      Value window() {
        return getDpsInitOperand(1)->get();
      }
      ShapedType getInputType() {
        return input().getType().cast<ShapedType>();
      }
      ShapedType getSrcType() {
        return src().getType().cast<ShapedType>();
      }
      ShapedType getIndiceType() {
        return indice().getType().cast<ShapedType>();
      }
      ShapedType getMaskType() {
        if (mask()) {
          return mask().getType().cast<ShapedType>();
        }
        return {};
      }
      int64_t getIndexDepth() {
        return getIndiceType().getShape().back();
      }
      LogicalResult reifyResultShapes(OpBuilder &b,
                                      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
        return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
      }
      MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for AtomicCAS
//===----------------------------------------------------------------------===//
def AtomicCASOp : Op<LinalgExt_Dialect, "atomic_cas", [
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     DestinationStyleOpInterface,
                     LinalgExtInterface,
                     ReifyRankedShapedTypeOpInterface]> {
  let summary = [{ LinalgExt atomic CAS operation for continuous buffer. }];
  let description = [{
    AtomicCASOp has three inputs: input, cmp and val.
    Compares cmp with input. if cmp == input, store val to input,
    else store the original value of input to init.
  }];

  let arguments = (ins
    Variadic<TensorOrMemref>:$inputs,
    TensorOrMemref:$init,
    LinalgExt_MemoryOrder:$memory_order
  );

  let results = (outs Variadic<TensorOrMemref>:$results);

  let assemblyFormat = [{
    $memory_order attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $init `:` type($init) `)`
    (`->` type($results)^)?
  }];

  let hasFolder = 1;
  let extraClassDeclaration = [{
    // Base helper functions.
    Value input() {
      return getDpsInputOperand(0)->get();
    }
    Value cmp() {
      return getDpsInputOperand(1)->get();
    }
    Value val() {
      return getDpsInputOperand(2)->get();
    }

    // Base helper function.
    ShapedType getInputType() {
      return input().getType().cast<ShapedType>();
    }

    ShapedType getCmpType() {
      return cmp().getType().cast<ShapedType>();
    }

    ShapedType getValType() {
      return val().getType().cast<ShapedType>();
    }

    ShapedType getInitType() {
      return getInit().getType().cast<ShapedType>();;
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }

    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// Op definition for GatherAtomicCAS
//===----------------------------------------------------------------------===//
def GatherAtomicCASOp : Op<LinalgExt_Dialect, "gather_atomic_cas", [
                           DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                           DestinationStyleOpInterface,
                           LinalgExtInterface,
                           ReifyRankedShapedTypeOpInterface]> {
  let summary = [{ LinalgExt atomic CAS operation for discrete buffer. }];
  let description = [{
    AtomicCASOp has four inputs: input, cmp, val and indice.
    Compares cmp with input. if cmp == input, store val to input,
    else store the original value of input to init.
    Note that the input must point to discrete data, so we add extra
    indice like ``GatherOp`` to deal with this situation.
  }];

  let arguments = (ins
    Variadic<TensorOrMemref>:$inputs,
    TensorOrMemref:$init,
    LinalgExt_MemoryOrder:$memory_order
  );

  let results = (outs Variadic<TensorOrMemref>:$results);

  let assemblyFormat = [{
    $memory_order attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $init `:` type($init) `)`
    (`->` type($results)^)?
  }];

  let hasFolder = 1;
  let extraClassDeclaration = [{
    // Base helper function.
    Value input() {
      return getDpsInputOperand(0)->get();
    }
    Value cmp() {
      return getDpsInputOperand(1)->get();
    }
    Value val() {
      return getDpsInputOperand(2)->get();
    }
    Value indice() {
      return getDpsInputOperand(3)->get();
    }

    // Base helper function.
    ShapedType getInputType() {
      return input().getType().cast<ShapedType>();
    }

    ShapedType getCmpType() {
      return cmp().getType().cast<ShapedType>();
    }

    ShapedType getValType() {
      return val().getType().cast<ShapedType>();
    }

    ShapedType getIndiceType() {
      return indice().getType().cast<ShapedType>();
    }

    ShapedType getInitType() {
      return getInit().getType().cast<ShapedType>();;
    }

    int64_t getIndexDepth() {
      return getIndiceType().getShape().back();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }

    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

def ExtYieldOp : LinalgExt_PureOp<"yield", [NoMemoryEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Op definition for PadOp
//===----------------------------------------------------------------------===//
def PadOp : LinalgExtBase_Op<"pad", [AttrSizedOperandSegments]> {
  let summary = [{ Pad operation. }];
  let description = [{
    Performs padding on the `input` by extending the dimensions with the `pvalue`.
    `low` and `high` are used to configure the padding size.

    The PadOp operation supports the following arguments:

    * input: the source tensor or memref on which to pad.
    * init: the result tensor or memref after padding.
    * pvalue: the value used to fill the tensor or memref.
    * low: A list contains the padding along the start of each dimension.
    * high: A list contains the padding along the end of each dimension.

    The result tensor or memref dimensions are `low` + `dim` + `high` along that
    dimension.
    The number of elements of `low` and `high` must match the rank of the input
    tensor or memref. They can be either a constant or a dynamic value.

    Example 1:
    ```
      %pad = linalg_ext.pad
        ins(%input:tensor<4x4xf32>)
        outs(%init:tensor<6x8xf32>)
        pvalue(0.0:f32)
        low = [1, 2]
        high = [1, 2] {} -> tensor<6x8xf32>
    ```
    Example 2:
    ```
      %pad = linalg_ext.pad
        ins(%input:tensor<4x4xf32>)
        outs(%init:tensor<6x?xf32>)
        pvalue(0.0:f32)
        low = [1, %arg1]
        high = [1, 2] {} -> tensor<6x?xf32>
    ```
  }];
  let arguments = (ins
      TensorOrMemref:$input,
      TensorOrMemref:$init,
      AnyType:$pvalue,
      Variadic<Index>:$low,
      Variadic<Index>:$high,
      DenseI64ArrayAttr:$static_low,
      DenseI64ArrayAttr:$static_high
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{attr-dict
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $init `:` type($init) `)`
      `pvalue` `(` $pvalue `:` type($pvalue) `)`
      `low` `=` custom<DynamicIndexList>($low, $static_low)
      `high` `=` custom<DynamicIndexList>($high, $static_high)
      $region (`->` type($result)^)?
    }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$init,
      "Value":$pvalue,
      "ArrayRef<OpFoldResult>":$low,
      "ArrayRef<OpFoldResult>":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Value":$input, "Value":$init,
      "Value":$pvalue,
      "ArrayRef<int64_t>":$low,
      "ArrayRef<int64_t>":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Value":$input, "Value":$init,
      "Value":$pvalue,
      "ValueRange":$low,
      "ValueRange":$high,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getDpsInputOperand(0)->get();
    }
    ShapedType getInputType() {
      return input().getType().cast<ShapedType>();
    }
    ShapedType getInitType() {
      return getInit().getType().cast<ShapedType>();
    }
    Type getPaddingValueType() {
      return getPvalue().getType();
    }
    //===----------------------------------------------------------------------===//
    // BEGIN copied from mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td
    //===----------------------------------------------------------------------===//
    // Return a vector of all the static or dynamic values (low/high padding) of
    // the op.
    inline SmallVector<OpFoldResult> getMixedPadImpl(ArrayRef<int64_t> staticAttrs,
                                                     ValueRange values) {
      Builder builder(*this);
      SmallVector<OpFoldResult> res;
      unsigned numDynamic = 0;
      unsigned count = staticAttrs.size();
      for (unsigned idx = 0; idx < count; ++idx) {
        if (ShapedType::isDynamic(staticAttrs[idx]))
          res.push_back(values[numDynamic++]);
        else
          res.push_back(builder.getI64IntegerAttr(staticAttrs[idx]));
      }
      return res;
    }
    SmallVector<OpFoldResult> getMixedLowPad() {
      return getMixedPadImpl(getStaticLow(), getLow());
    }
    SmallVector<OpFoldResult> getMixedHighPad() {
      return getMixedPadImpl(getStaticHigh(), getHigh());
    }
    // Return true if low padding is guaranteed to be 0.
    bool hasZeroLowPad() {
      return llvm::all_of(getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
    // Return true if high padding is guaranteed to be 0.
    bool hasZeroHighPad() {
      return llvm::all_of(getMixedHighPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      });
    }
    // Infer the shape of the result tensor given the type of the input tensor
    // and paddings.
    static RankedTensorType inferResultType(RankedTensorType inputType,
                                            ArrayRef<int64_t> staticLow,
                                            ArrayRef<int64_t> staticHigh,
                                            ArrayRef<int64_t> resultShape = {});
    // Return the pad value if it is a constant. Return null value otherwise.
    Value getConstantPaddingValue();
    // Return the dimensions with a non-zero low or high padding.
    llvm::SmallBitVector getPaddedDims();
    //===----------------------------------------------------------------------===//
    // END copied from mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td
    //===----------------------------------------------------------------------===//
    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// Op definition for AssertOp
//===----------------------------------------------------------------------===//
def AssertOp : Op<LinalgExt_Dialect, "assert", [
                  DestinationStyleOpInterface,
                  LinalgExtInterface,
                  ReifyRankedShapedTypeOpInterface]> {
  let summary = "lianlg assert operation";
  let description = [{
    `linalg_ext.assert` takes a condition tensor, a message string.
    If the condition is false, the message is printed, and the program is aborted.
  }];

  let arguments = (ins
    TensorOrMemref:$condition,
    StrAttr:$msg
  );
  let results = (outs Variadic<AnyTensor>:$results);

  let assemblyFormat = [{
    attr-dict `ins` `(` $condition `:` type($condition) `)`
    (`->` type($results)^)?

  }];

  let extraClassDeclaration =  [{

    ShapedType getInitType() {
      return getCondition().getType().cast<ShapedType>();;
    }
    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      std::pair<unsigned, unsigned> outputsIndexAndLength =
          getODSOperandIndexAndLength(1);
      return std::make_pair<int64_t, int64_t>(
          outputsIndexAndLength.first,
          outputsIndexAndLength.first + outputsIndexAndLength.second);
    }
    MutableOperandRange getDpsInitsMutable() { return getConditionMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// Scan Op.
//===----------------------------------------------------------------------===//
def ScanOp : LinalgExtBase_Op<"scan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    SingleBlockImplicitTerminator<"::mlir::triton::linalg_ext::ExtYieldOp">,
    AttrSizedOperandSegments]> {
  let summary = "Scan Operations";
 let description = [{
    Applies the `combiner` to each elements with
    a carry in `inputs` along the provided `dimensions`
    and update the carry.

    math:
      output[0] = fn(input[0], init)
      output[i] = fn(input[i], output[i - 1]), i > 0

    Example:
    ```
      %scanned = linalg_ext.scan
          ins(%input:tensor<16x32x64xf32>)
          outs(%output, %init: tensor<16x32x64xf32>, tensor<16x64xf32>)
          dimension = [1]
          {
          ^bb0(%in: f32, %out: f32, %init: f32):
            %0 = arith.addf %init, %in: f32
            linalg_ext.yield %0, %0: f32, f32
          }
    ```
  }];
  let arguments = (ins
    // Input arg
    Variadic<TensorOrMemref>:$inputs,
    // Output arg and init arg
    Variadic<TensorOrMemref>:$inits,
    // Dimension arg
    DenseI64ArrayAttr:$dimensions,
    // Reverse arg
    BoolAttr:$reverse
  );
  let results = (outs Variadic<TensorOrMemref>:$results);
  let regions = (region SizedRegion<1>:$combiner);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$inits,
      "ArrayRef<int64_t>":$dimensions, "bool":$reverse,
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $inits `:` type($inits) `)`
    `dimensions` `=` $dimensions
    `reverse` `=` $reverse
    $combiner (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    Block::BlockArgListType getRegionInputArgs() {
      return getBlock()->getArguments().take_front(
          cast<DestinationStyleOpInterface>(*this->getOperation())
              .getNumDpsInputs());
    }
    Block::BlockArgListType getRegionOutputArgs() {
      return getBlock()->getArguments().take_back(
          cast<DestinationStyleOpInterface>(*this->getOperation())
              .getNumDpsInits());
    }
    llvm::SmallVector<Value> inputs() {
      return llvm::to_vector(
          llvm::map_range(getDpsInputOperands(),
                          [&](OpOperand *operand) { return operand->get(); }));
    }
    llvm::SmallVector<Value> inits() {
      llvm::SmallVector<Value> inits;
      int64_t numInits = getNumDpsInits() / 2;
      for (int64_t i = 0; i < numInits; ++i) {
        inits.push_back(getDpsInitOperand(i + numInits)->get());
      }
      return inits;
    }
    llvm::SmallVector<Value> outputs() {
      llvm::SmallVector<Value> outputs;
      int64_t numOutputs = getNumDpsInits() / 2;
      for (int64_t i = 0; i < numOutputs; ++i) {
        outputs.push_back(getDpsInitOperand(i)->get());
      }
      return outputs;
    }
    ShapedType getOperandType() {
      return inputs()[0].getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }

    SmallVector<OpOperand *> getOpOperandsMatchingBBargs() {
      // Latest llvm use getDpsInputs, change it when upgrade.
      return getDpsInputOperands();
    }

    Block* getBlock() { return getBody(); }

    MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LibdeviceCall op.
//===----------------------------------------------------------------------===//
def LibdeviceCallOp : Op<LinalgExt_Dialect, "libdevice_call", [
                         DestinationStyleOpInterface,
                         LinalgExtInterface,
                         ReifyRankedShapedTypeOpInterface]> {
  let summary = "linalg libdevice_call operation";
  let description = [{
    LibdeviceCallOp is used to undertake operators that need to call operators from the libdevice library.
    It is generally used to undertake the TT_ExtElemwiseOp from triton.

    Example:
    ```
      %add = linalg_ext.libdevice_call
             ins(%arg0, %arg1 : tensor<16x10x10x64xi32>, tensor<16x10x10x64xi32>)
             outs(%fill : tensor<16x10x10x64xi32>)
             symbol = "__vector_add_s32" -> tensor<16x10x10x64xi32>
    ```
  }];

  let arguments = (ins
    // Input args
    Variadic<AnyType>:$inputs,
    // Output arg
    TensorOrMemref:$init,
    // function name
    StrAttr:$symbol
  );

  let results = (outs Variadic<TensorOrMemref>:$results);
  let assemblyFormat = [{
      attr-dict
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `outs` `(` $init `:` type($init) `)`
      `symbol` `=` $symbol
       (`->` type($results)^)?
    }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
        "StringAttr":$symbol, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
    OpBuilder<(ins "ValueRange":$inputs, "Value":$init,
        "StringRef":$symbol, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    llvm::SmallVector<Value> inputs() {
      return llvm::to_vector(
          llvm::map_range(getDpsInputOperands(),
                          [&](OpOperand *operand) { return operand->get(); }));
    }
    ShapedType getOperandType() {
      return getInputs()[0].getType().cast<ShapedType>();
    }
    ShapedType getInitType() {
      return getInit().getType().cast<ShapedType>();;
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgExtOp>(getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }

    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScalarLibdeviceCall op.
//===----------------------------------------------------------------------===//
def ScalarLibdeviceCallOp : LinalgExt_PureOp<"scalar_libdevice_call", [NoMemoryEffect]> {
  let summary = "LinalgExt scalar_libdevice_call op";
  let description = [{
    `linalg_ext.scalar_libdevice_call` is used to undertake operators that need to call
    operators from the libdevice library which inputs and result type are both scalar.
    It is generally used to undertake the TT_ExtElemwiseOp from triton.
  Example:
    ```
      %add = linalg_ext.scalar_libdevice_call
             ins(%arg0, %arg1 : i32, i32)
             symbol = "__cn_scalar_add_s32" -> i32
    ```
  }];

  let arguments = (ins
    // Input args
    Variadic<AnyType>:$inputs,
    // function name
    StrAttr:$symbol
  );

  let results = (outs AnyType:$result);
  let assemblyFormat = [{
      attr-dict
      (`ins` `(` $inputs^ `:` type($inputs) `)`)?
      `symbol` `=` $symbol
      `->` type($result)
    }];
  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "ValueRange":$inputs,
        "StringAttr":$symbol, CArg<"ArrayRef<NamedAttribute>",
        "{}">:$attributes)>,
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
#endif // TRITON_LINALG_DIALECT_LINALGEXT_IR_LINALGEXTOPS_TD
