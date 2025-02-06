// RUN: triton-linalg-opt -convert-math-to-linalg -split-input-file %s | FileCheck %s

// -----
func.func @math_log(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.log } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.log %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_exp(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.exp } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.exp %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_log_scalar(%arg0: f32) {
  // CHECK: math.log %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.log %arg0 : f32
  return
}

// -----
func.func @math_log_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.log } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.log %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_log_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.log } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.log %arg0 : tensor<?x?xf32>
  return
}

// -----
func.func @math_exp_scalar(%arg0: f32) {
  // CHECK: math.exp %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.exp %arg0 : f32
  return
}

// -----
func.func @math_exp_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.exp } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.exp %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_exp_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.exp } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.exp %arg0 : tensor<?x?xf32>
  return
}

// -----
func.func @math_cos_scalar(%arg0: f32) {
  // CHECK: math.cos %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.cos %arg0 : f32
  return
}

// -----
func.func @math_cos_tensor_static(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.cos } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.cos %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_cos_tensor_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.cos } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.cos %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_cos_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.cos } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.cos %arg0 : tensor<?x?xf32>
  return
}

// -----
func.func @math_sin_scalar(%arg0: f32) {
  // CHECK: math.sin %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.sin %arg0 : f32
  return
}

// -----
func.func @math_sin_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sin } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.sin %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_sin_tensor_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sin } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.sin %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_sin_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sin } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.sin %arg0 : tensor<?x?xf32>
  return
}


// -----
func.func @math_sqrt_scalar(%arg0: f32) {
  // CHECK: math.sqrt %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.sqrt %arg0 : f32
  return
}

// -----
func.func @math_sqrt_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sqrt } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.sqrt %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_sqrt_tensor_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sqrt } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.sqrt %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_sqrt_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.sqrt } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.sqrt %arg0 : tensor<?x?xf32>
  return
}

// -----
func.func @math_absi_scalar(%arg0: i32) {
  // CHECK: math.absi %arg0 : i32
  // CHECK-NOT: linalg.map
  %0 = math.absi %arg0 : i32
  return
}

// -----
func.func @math_absi_tensor_staic(%arg0: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absi } ins(%arg0 : tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = math.absi %arg0 : tensor<128xi32>
  return
}

// -----
func.func @math_absi_tensor_partial_static(%arg0: tensor<128x?xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xi32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absi } ins(%arg0 : tensor<128x?xi32>) outs(%[[INIT]] : tensor<128x?xi32>)
  %0 = math.absi %arg0 : tensor<128x?xi32>
  return
}

// -----
func.func @math_absi_tensor_dynamic(%arg0: tensor<?x?xi32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xi32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xi32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absi } ins(%arg0 : tensor<?x?xi32>) outs(%[[INIT]] : tensor<?x?xi32>)
  %0 = math.absi %arg0 : tensor<?x?xi32>
  return
}

// -----
func.func @math_absf_scalar(%arg0: f32) {
  // CHECK: math.absf %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.absf %arg0 : f32
  return
}

// -----
func.func @math_absf_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absf } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.absf %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_absf_tensor_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absf } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.absf %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_absf_tensor_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[DYNAMIC_D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_D0]], %[[DYNAMIC_D1]]) : tensor<?x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.absf } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
  %0 = math.absf %arg0 : tensor<?x?xf32>
  return
}

// -----
func.func @math_trunc_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.trunc } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.trunc %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_round_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.round } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.round %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_floor_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.floor } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.floor %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_ceil_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.ceil } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.ceil %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_log2_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.log2 } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.log2 %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_exp2_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.exp2 } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.exp2 %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_fma_tensor_staic(%arg0: tensor<100x10xf32>, %arg1: tensor<100x10xf32>) -> tensor<100x10xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<100x10xf32>
  // CHECK: %mapped = linalg.map { math.fma } ins(%arg0, %arg0, %arg1 : tensor<100x10xf32>, tensor<100x10xf32>, tensor<100x10xf32>) outs(%[[INIT]] : tensor<100x10xf32>)
  %0 = math.fma %arg0, %arg0, %arg1: tensor<100x10xf32>
  return %0 : tensor<100x10xf32>
}

// -----
func.func @math_rsqrt_tensor_staic(%arg0: tensor<64x128xf16>) -> tensor<64x128xf16> {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<64x128xf16>
  // CHECK: %mapped = linalg.map { math.rsqrt } ins(%arg0 : tensor<64x128xf16>) outs(%[[INIT]] : tensor<64x128xf16>)
  %0 = math.rsqrt %arg0: tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// -----
tt.func @tt_mulhiui_vector_i32(%arg0: tensor<16x16xi32>, %arg1: tensor<16x16xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<16x16xi32>
  // CHECK: %mapped = linalg.map { math_ext.mulhiui } ins(%arg0, %arg1 : tensor<16x16xi32>, tensor<16x16xi32>) outs(%[[INIT]] : tensor<16x16xi32>)
  %0 = math_ext.mulhiui %arg0, %arg1 : tensor<16x16xi32>
  tt.return
}

// -----
func.func @math_tanh_scalar(%arg0: f32) {
  // CHECK: math.tanh %arg0 : f32
  // CHECK-NOT: linalg.map
  %0 = math.tanh %arg0 : f32
  return
}

// -----
func.func @math_tanh_tensor_staic(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.tanh } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.tanh %arg0 : tensor<128xf32>
  return
}

// -----
func.func @math_tanh_tensor_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.tanh } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.tanh %arg0 : tensor<128x?xf32>
  return
}

// -----
func.func @math_powf_scalar(%arg0: f32, %arg1: f32) {
  // CHECK: math.powf %arg0, %arg1 : f32
  // CHECK-NOT: linalg.map
  %0 = math.powf %arg0, %arg1 : f32
  return
}

// -----
func.func @math_fpowi_scalar(%arg0: f32, %arg1: i32) {
  // CHECK: math.fpowi %arg0, %arg1 : f32, i32
  // CHECK-NOT: linalg.map
  %0 = math.fpowi %arg0, %arg1 : f32, i32
  return
}

// -----
func.func @math_powf_tensor_staic(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.powf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.powf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @math_fpowi_tensor_staic(%arg0: tensor<128xf32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.fpowi } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = math.fpowi %arg0, %arg1 : tensor<128xf32>, tensor<128xi32>
  return
}

// -----
func.func @math_powf_tensor_partial_static(%arg0: tensor<128x?xf32>, %arg1: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.powf } ins(%arg0, %arg1 : tensor<128x?xf32>, tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.powf %arg0, %arg1 : tensor<128x?xf32>
  return
}

// -----
func.func @math_fpowi_tensor_partial_static(%arg0: tensor<128x?xf32>, %arg1: tensor<128x?xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { math.fpowi } ins(%arg0, %arg1 : tensor<128x?xf32>, tensor<128x?xi32>) outs(%[[INIT]] : tensor<128x?xf32>)
  %0 = math.fpowi %arg0, %arg1 : tensor<128x?xf32>, tensor<128x?xi32>
  return
}
