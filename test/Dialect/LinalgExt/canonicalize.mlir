// RUN: triton-linalg-opt %s -canonicalize -split-input-file | FileCheck %s

func.func @foldPadOpWithConstPaddingSize(%input: tensor<4x6x8xf32>, %init: tensor<8x?x16xf32>, %size : index) -> tensor<8x?x16xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %pad = linalg_ext.pad ins(%input: tensor<4x6x8xf32>) outs(%init: tensor<8x?x16xf32>) pvalue(%cst: f32) low = [%c1, %c0, 4] high = [%c3, %size, %c4] {
    ^bb0(%arg0 :index):
      linalg_ext.yield %arg0 : index
    } -> tensor<8x?x16xf32>
  return %pad : tensor<8x?x16xf32>
}
// CHECK: linalg_ext.pad ins(%[[ARG0:.*]] : tensor<4x6x8xf32>) outs(%[[ARG1:.*]] : tensor<8x?x16xf32>) pvalue(%[[CST:.*]] : f32) low = [1, 0, 4] high = [3, %[[ARG2:.*]], 4]

// -----
func.func @foldPadOpWithZeroPaddingSize(%input: tensor<4x6x8xf32>, %init: tensor<4x6x8xf32>) -> tensor<4x6x8xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %pad = linalg_ext.pad ins(%input: tensor<4x6x8xf32>) outs(%init: tensor<4x6x8xf32>) pvalue(%cst: f32) low = [%c0, %c0, 0] high = [0, %c0, %c0] {
    ^bb0(%arg0 :index):
      linalg_ext.yield %arg0 : index
    } -> tensor<4x6x8xf32>
  return %pad : tensor<4x6x8xf32>
}
// CHECK-NOT: linalg_ext.pad 
// CHECK: return %[[ARG0:.*]] : tensor<4x6x8xf32>