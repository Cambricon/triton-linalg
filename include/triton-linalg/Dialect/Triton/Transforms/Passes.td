//===- Passes.td - Transforms pass definition file ---------*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for triton passes.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_TD
#define TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def CanonicalizeTriton : Pass<"canonicalize-triton", "::mlir::ModuleOp"> {
  let summary = "Canonicalize triton IR.";
  let description = [{
    Canonicalize triton IR, for example, transform broadcast to its standard use case.
  }];
  let constructor = "mlir::triton::createCanonicalizeTritonPass()";
  let dependentDialects = [
      "triton::TritonDialect"
  ];
}

def ExtractLikeMoveBackwardPass : Pass<"extract-like-move-backward", "::mlir::func::FuncOp"> {
  let summary = "Move backward tensor.extract and tensor.extract_slice operations.";
  let description = [{
    This pass moves extract-like operations backward, and changes
    operations in the backward path with extracted tensor or scalar.
    Currently, tensor.extract and tensor.extract_slice are supported.

    For example, consider the following examples:

    Case 1.
    ``` mlir
    func.func @extract(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
      %c0 = arith.constant 0 : index
      %0 = arith.addi %arg0, %arg1 : tensor<128xi32>
      %1 = tensor.extract %0[%c0] : tensor<128xi32>
      return %1 : i32
    }
    ```

    After running, we get the expected:

    ``` mlir
    func.func @extract(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
      %c0 = arith.constant 0 : index
      %0 = tensor.extract %arg0[%c0] : tensor<128xi32>
      %1 = tensor.extract %arg1[%c0] : tensor<128xi32>
      %2 = arith.addi %0, %1 : i32
      return %2 : i32
    }
    ```

    Case 2.
    ``` mlir
    func.func @extract_slice(%arg0: tensor<128x16xi32>,
                             %arg1: tensor<128x16xi32>) -> tensor<16x4xi32> {
      %0 = tensor.empty() : tensor<128x16xi32>
      %1 = linalg.map { arith.addi }
           ins(%arg0, %arg1 : tensor<128x16xi32>, tensor<128x16xi32>)
           outs(%0 : tensor<128x16xi32>)
      %2 = tensor.extract_slice %1[0, 0] [16, 4] [2, 2]
           : tensor<128x16xi32> to tensor<16x4xi32>
      return %2 : tensor<16x4xi32>
    }
    ```

    After running, we get the expected:

    ``` mlir
    func.func @extract_slice(%arg0: tensor<128x16xi32>,
                             %arg1: tensor<128x16xi32>) -> tensor<16x4xi32> {
      %extracted_slice = tensor.extract_slice %arg0[0, 0] [16, 4] [2, 2]
                         : tensor<128x16xi32> to tensor<16x4xi32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [16, 4] [2, 2]
                           : tensor<128x16xi32> to tensor<16x4xi32>
      %0 = tensor.empty() : tensor<16x4xi32>
      %mapped = linalg.map { arith.addi }
                ins(%extracted_slice, %extracted_slice_0
                    : tensor<16x4xi32>, tensor<16x4xi32>)
                outs(%0 : tensor<16x4xi32>)
      return %mapped : tensor<16x4xi32>
    }
    ```
  }];
  let constructor = "mlir::triton::createExtractLikeMoveBackwardPass()";
}

def PointerStrengthReductionPtr : Pass<"ptr-strength-reduction", "::mlir::ModuleOp"> {
  let summary = "Canonicalize triton operations with pointer to operations with offsets.";
  let description = [{
    This pass standardizes computations involving Triton pointers and streamlines Triton IR,
    targeting operations such as 'addptr', 'broadcast', and others.  It also optimizes
    various IR constructs (e.g., control flow mechanisms) that utilize Triton pointer operands.

    For example, consider the following examples:

    Case 1.
    ``` mlir
    tt.func @test(%arg0: !tt.ptr<f32>, %arg1: tensor<1x64xi32>) -> tensor<32x64xf32> {
      %0 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1x64x!tt.ptr<f32>>
      %1 = tt.addptr %0, %arg1 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
      %2 = tt.broadcast %1 : (tensor<1x64x!tt.ptr<f32>>) -> tensor<32x64x!tt.ptr<f32>>
      %3 = tt.load %2 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<32x64xf32>
      tt.return %3 : tensor<32x64xf32>
    }
    ```

    After running, we get the expected:

    ``` mlir
    tt.func @test(%arg0: !tt.ptr<f32>, %arg1: tensor<1x64xi32>) -> tensor<32x64xf32> {
      %0 = tt.broadcast %arg1 : (tensor<1x64xi32>) -> tensor<32x64xi32>
      %1 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x64x!tt.ptr<f32>>
      %2 = tt.addptr %1, %0 : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
      %3 = tt.load %2 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<32x64xf32>
      tt.return %3 : tensor<32x64xf32>
    }
    ```

    Case 2.
    ``` mlir
    %7 = scf.for %arg1 = %3 to %4 step %5 iter_args(%arg3 = %2) -> (tensor<32x!tt.ptr<f32>>) {
      %9 = arith.index_cast %arg1 : index to i32
      %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %11 = tt.splat %9 : (i32) -> tensor<32xi32>
      %12 = arith.addi %10, %11 : tensor<32xi32>
      %13 = tt.addptr %arg3, %12 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      scf.yield %13 : tensor<32x!tt.ptr<f32>>
    }
    ```

    After running, we get the expected:

    ``` mlir
    %1 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %0) -> (tensor<32xi32>) {
      %5 = arith.index_cast %arg1 : index to i32
      %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %7 = tt.splat %5 : (i32) -> tensor<32xi32>
      %8 = arith.addi %6, %7 : tensor<32xi32>
      %9 = arith.addi %8, %arg2 : tensor<32xi32>
      scf.yield %9 : tensor<32xi32>
    }
    ```
  }];
  let constructor = "mlir::triton::createPointerStrengthReductionPass()";
  let dependentDialects = [
    "triton::TritonDialect","scf::SCFDialect"
  ];
}

def WrapFuncBodyWithSingleBlock : Pass<"wrap-func-body-with-single-block", "::mlir::ModuleOp"> {
  let summary = "Wrap function body with single block";
  let description = [{
    This pass wraps function body into a block by moving body to a `scf.execute_region`.
  }];

  let constructor = "mlir::triton::createWrapFuncBodyWithSingleBlockPass()";

  let dependentDialects = ["triton::TritonDialect", "scf::SCFDialect"];
}
#endif // TRITON_LINALG_DIALECT_TRITON_TRANSFORMS_PASSES_TD
