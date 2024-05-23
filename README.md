# triton-linalg

A shared middle-layer for the Triton Compiler. Currently, it has successfully supported the Cambrian backend as a front-end representation,
and functionally, it is capable of handling nearly all features of the Triton language.

In the era of artificial intelligence, numerous domain-specific architectures (DSA) have emerged to meet performance demands. 
To optimize the efficiency of program executed on these architectures, designers often define a series of coarse-grained operators. 
These operators are tailor-made for specific tasks, enabling the hardware to handle complex computational
demands more efficiently. For this purpose, the `triton-linalg` repository adheres to several principles during the conversion process:

- Use structured operators as much as possible, avoiding the use of `linalg.generic`, For instance:
   + `tt.reduce` becomes `linalg.reduce` and `tt.dot` becomes `linalg.matmul`.
   + element-wise `arith` and `math` operators are converted to their corresponding `linalg.map` versions.
- Identify operator semantics as early as possible. For instance: `tt.reduce` becomes `linalg.pool`.

## Usage

This repo now includes `triton` as a submodule and builds as an out-of-tree backend.

To build this repo clone `triton-linalg` to a folder called `triton_linalg` (notice the **underscore**).

You need to set the `TRITON_PLUGIN_DIRS` environment variable to the location of your `triton-linalg` directory for `triton` to find it.

```
export TRITON_PLUGIN_DIRS=$(pwd)/triton_linalg

git clone --recurse-submodules https://github.com/Cambricon/triton-linalg.git
cd triton-linalg/triton
```

To build with Clang:

```
python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist
sudo apt-get update -y
sudo apt-get install -y ccache clang lld
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv '.[tests]'
```

To build with a virtualenv:

```
python3 -m venv .venv --prompt triton
source .venv/bin/activate

pip3 install ninja cmake wheel pytest
pip3 install -e python --no-build-isolation
```

### Stand-Alone
Linalg can be used as a stand-alone component to convert Triton dialect to the Linalg dialect. This is intended for testing and validation purposes, but could potentially be used before sending the IR to another MLIR complier.

Stand-alone example:

```
triton-linalg-opt --convert-triton-to-linalg ${TRITON_LINALG_DIR}/test/Dialect/LinalgExt/ops.mlir
```

## Implementation details

`triton-linalg` is composed of four parts: Analysis, Conversion, Dialect, and Transforms.

- Dialect: Defines Linalg Extension operators as well as Auxiliary support operators.
- Analysis: Includes algorithms for pointer analysis, Mask Tracker, and pointer offset Tracker.
- Conversion: Contains conversions from triton/arith/math to linalg.
- Transforms: Expands the standardization of community Dialects and includes some auxiliary optimization passes.

### Dialect

`triton-linalg` has added two new dialects.

- Auxiliary: Includes some auxiliary operators, such as the `view` operator, used to bind structured information such as offsets, sizes, and strides to a llvm pointer, converting them into memref types.

```
aux.view %ptr to offset: [0], sizes: [%size0, 10], strides: [1, %stride1]
    : llvm.ptr<f32> to memref<?x10xf32, strided<[1, ?], offset: 0>>
```

- LinalgExt: Includes extension operators of the Linalg Dialect, such as `linalg_ext.gather`, `linalg_ext.scatter`.


### Analysis

As part of the conversion process, there are three important analyses:

1. AxisInfo analysis:
   + This analysis is based on the official `AxisInfoAnalysis`, it has been expanded to support the recognition of multi-dimensional continuity.

2. Mask Tracker:
   + This analysis is responsible for handling masked loads and stores.

3. Pointer Offset Tracker.
   + This analysis is responsible for tracking pointer offsets.

### Conversion

We introduce the `TritonToLinalg` conversion pass that converts the `triton` dialects to the `linalg` dialect on *tensors*. 
This means the resulting IR is fully compatible with `linalg` tiling and fusion transformation passes. 
As mentioned in the `Pointer analysis`'s description, we do however have to deal with memref instructions at the
load and store boundaries and have to convert them to tensors using `bufferization.to_tensor`. 
Here's a simple example of what the IR looks like:

<a id="add_kernel"></a>

```mlir

module attributes {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

```

after conversion:

```mlir
module {
  func.func public @add_kernel_01234(%arg0: int64, %arg1: int64, %arg2: int64, %arg3: i32) {
    ...
    %offset = ...
    %extracted = tensor.extract %offset[%c0] : tensor<1024xi32>
    %offset0 = arith.index_cast %extracted : i32 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %view_memref = aux.view %ptr to offset: [%offset0], sizes: [%size], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %tensor = bufferization.to_tensor %view_memref restrict writable : memref<?xf32, #map>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %35, %c0 : tensor<?xf32>
    %empty = tensor.empty(%dim) : tensor<?xf32>
    %data = linalg.copy ins(%tensor : tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    ...
    return
  }
}

```

Important details to note is the handling of pointers(not block pointers) for `tt.load` and `tt.store` operations.
The pointers are mainly categorized into continuous(pointer to a tensor) and discrete cases.

1. Continuous case: 
   + Based on the pointer tensor information obtained from `AxisInfoAnalysis`, combined with the offsets tracked by the `Pointer Offsets Tracker`, the actual pointer offsets/strides can be calculated. Then, using `aux.view`, `buffertion.to_tensor` and `linalg.copy`, it is converted to tensor semantics.
2 Discrete case: 
   + We cannot deduce continuity, nor can we determine the actual size of the space pointed to by the pointer. Therefore, we assume that the pointer points to an infinitely large space. Then using `aux.view`, `buffertion.to_tensor` and `linalg_ext.gather`, it is converted to tensor semantics.

### Transforms

We have introduced some optimization passes to assist in the conversion process, there are some important passes:

1. WrapFuncBodyWithSingleBlock:
   + This pass wraps function body into a block by moving body to a `scf.execute_region`. The primary aim is to resolve the issue of inlining of function calls within `scf.for` due to multiple block problems.
2. ExtractLikeMoveBackwardPass：
   + This pass moves extract-like operations backward, and changes operations in the backward path with extracted tensor or scalar.

### Pass Pipeline

We introduce the `triton-to-linalg` pass pipeline to convert a triton dialect ir to linalg/linalg_ext dialect ir. Here is an example:

Regarding the IR of [add_kernel](#add_kernel) in the Conversion section above, the result after executing the triton-to-linalg pipeline is as follows.

```
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @add_kernel(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.maxsi %4, %2 : index
    %6 = arith.minsi %3, %5 : index
    %7 = arith.subi %6, %2 : index
    %8 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %view_memref = aux.view %8 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %9 = bufferization.to_tensor %view_memref restrict writable : memref<?xf32, #map>
    %10 = tensor.empty(%7) : tensor<?xf32>
    %11 = linalg.copy ins(%9 : tensor<?xf32>) outs(%10 : tensor<?xf32>) -> tensor<?xf32>
    %12 = tensor.empty() : tensor<1024xf32>
    %13 = arith.subi %c1024, %7 : index
    %14 = linalg_ext.pad ins(%11 : tensor<?xf32>) outs(%12 : tensor<1024xf32>) pvalue(%cst : f32) low = [0] high = [%13] {
    ^bb0(%arg4: f32):
      linalg_ext.yield %arg4 : f32
    } -> tensor<1024xf32>
    %15 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %view_memref_0 = aux.view %15 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %16 = bufferization.to_tensor %view_memref_0 restrict writable : memref<?xf32, #map>
    %17 = linalg.copy ins(%16 : tensor<?xf32>) outs(%10 : tensor<?xf32>) -> tensor<?xf32>
    %18 = linalg_ext.pad ins(%17 : tensor<?xf32>) outs(%12 : tensor<1024xf32>) pvalue(%cst : f32) low = [0] high = [%13] {
    ^bb0(%arg4: f32):
      linalg_ext.yield %arg4 : f32
    } -> tensor<1024xf32>
    %mapped = linalg.map { arith.addf } ins(%14, %18 : tensor<1024xf32>, tensor<1024xf32>) outs(%12 : tensor<1024xf32>)
    %19 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %view_memref_1 = aux.view %19 to offset: [%2], sizes: [%7], strides: [1] : !llvm.ptr to memref<?xf32, #map>
    %extracted_slice = tensor.extract_slice %mapped[0] [%7] [1] : tensor<1024xf32> to tensor<?xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %view_memref_1 : (tensor<?xf32>, memref<?xf32, #map>) -> ()
    return
  }
}
```

## Related work

Its approach is similar to Microsoft's [triton-shared](https://github.com/microsoft/triton-shared.git),
the main difference is the pointer analysis.
