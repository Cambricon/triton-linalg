// RUN: triton-linalg-opt --convert-triton-to-tensor %s -split-input-file | FileCheck %s

// CHECK: tensor.insert_slice %arg0
// CHECK: tensor.insert_slice %arg1
tt.func public @cat(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) {
  %0 = tt.cat %arg0, %arg1 : tensor<32xf32> -> tensor<64xf32>
  tt.return
}
