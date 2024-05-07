// RUN: triton-linalg-opt --canonicalize-triton %s -split-input-file| FileCheck %s

// CHECK-LABEL: func.func @broadcast_0d
// CHECK-NEXT: tt.splat %arg0 : tensor<i32> -> tensor<256xi32>
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
func.func @broadcast_0d(%arg0: tensor<i32>) -> tensor<256xi32> {
  %0 = tt.broadcast %arg0 : tensor<i32> -> tensor<256xi32>
  return %0 : tensor<256xi32>
}

// -----

// CHECK-LABEL: func.func @broadcast_0d_to_3d
// CHECK-NEXT: tt.splat %arg0 : tensor<i32> -> tensor<4x4x256xi32>
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
func.func @broadcast_0d_to_3d(%arg0: tensor<i32>) -> tensor<4x4x256xi32> {
  %0 = tt.broadcast %arg0 : tensor<i32> -> tensor<4x4x256xi32>
  return %0 : tensor<4x4x256xi32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_1dto2d_1
// CHECK-NEXT: %[[CST:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast %[[CST]] : tensor<1x8xf32> -> tensor<2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_1dto2d_1(%arg0: tensor<8xf32>) -> tensor<2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_2dto3d_2
// CHECK-NEXT: %[[CST:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<2x8xf32> -> tensor<1x2x8xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast %[[CST:.*]] : tensor<1x2x8xf32> -> tensor<2x2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_2dto3d_2(%arg0: tensor<2x8xf32>) -> tensor<2x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<2x8xf32> -> tensor<2x2x8xf32>
  return %0 : tensor<2x2x8xf32>
}

// -----
// CHECK-LABEL: func.func @broadcast_tensor_2dto3d_3
// CHECK-NEXT: %[[CST:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<1x8xf32> -> tensor<1x1x8xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast %[[CST:.*]] : tensor<1x1x8xf32> -> tensor<4x2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_2dto3d_3(%arg0: tensor<1x8xf32>) -> tensor<4x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<1x8xf32> -> tensor<4x2x8xf32>
  return %0 : tensor<4x2x8xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_2dto4d_1
// CHECK-NEXT: %[[EXP1:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<2x4xf32> -> tensor<1x2x4xf32>
// CHECK-NEXT: %[[EXP2:.*]] = tt.expand_dims %[[EXP1]] {axis = 0 : i32} : tensor<1x2x4xf32> -> tensor<1x1x2x4xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast  %[[EXP2]] : tensor<1x1x2x4xf32> -> tensor<2x8x2x4xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_2dto4d_1(%arg0: tensor<2x4xf32>) -> tensor<2x8x2x4xf32> {
  %0 = tt.broadcast %arg0 : tensor<2x4xf32> -> tensor<2x8x2x4xf32>
  return %0 : tensor<2x8x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_2dto4d_2
// CHECK-NEXT: %[[EXP1:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<1x8xf32> -> tensor<1x1x8xf32>
// CHECK-NEXT: %[[EXP2:.*]] = tt.expand_dims %[[EXP1]] {axis = 0 : i32} : tensor<1x1x8xf32> -> tensor<1x1x1x8xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast  %[[EXP2]] : tensor<1x1x1x8xf32> -> tensor<16x4x2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_2dto4d_2(%arg0: tensor<1x8xf32>) -> tensor<16x4x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<1x8xf32> -> tensor<16x4x2x8xf32>
  return %0 : tensor<16x4x2x8xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_2dto5d
// CHECK-NEXT: %[[EXP1:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8x8xf32> -> tensor<1x8x8xf32>
// CHECK-NEXT: %[[EXP2:.*]] = tt.expand_dims %[[EXP1]] {axis = 0 : i32} : tensor<1x8x8xf32> -> tensor<1x1x8x8xf32>
// CHECK-NEXT: %[[EXP3:.*]] = tt.expand_dims %[[EXP2]] {axis = 0 : i32} : tensor<1x1x8x8xf32> -> tensor<1x1x1x8x8xf32>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast %[[EXP3]]  : tensor<1x1x1x8x8xf32> -> tensor<8x8x8x8x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_2dto5d(%arg0: tensor<8x8xf32>) -> tensor<8x8x8x8x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<8x8xf32> -> tensor<8x8x8x8x8xf32>
  return %0 : tensor<8x8x8x8x8xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_shold_not_be_converted_1
// CHECK-NEXT: tt.broadcast %arg0 : tensor<2x1x8xf32> -> tensor<2x2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_shold_not_be_converted_1(%arg0: tensor<2x1x8xf32>) -> tensor<2x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<2x1x8xf32> -> tensor<2x2x8xf32>
  return %0 : tensor<2x2x8xf32>
}

// -----

// CHECK-LABEL: func.func @broadcast_tensor_shold_not_be_converted_2
// CHECK-NEXT: tt.broadcast %arg0 : tensor<2x1x1x8xf32> -> tensor<2x2x2x8xf32>
// CHECK-NOT: tt.splat
func.func @broadcast_tensor_shold_not_be_converted_2(%arg0: tensor<2x1x1x8xf32>) -> tensor<2x2x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<2x1x1x8xf32> -> tensor<2x2x2x8xf32>
  return %0 : tensor<2x2x2x8xf32>
}

// -----

// CHECK-LABEL: @broadcast_ptr_tensor
// CHECK-NEXT: %[[CST:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<2x8x!tt.ptr<f32>> -> tensor<1x2x8x!tt.ptr<f32>>
// CHECK-NEXT: %[[RET:.*]] = tt.broadcast %[[CST]] : tensor<1x2x8x!tt.ptr<f32>> -> tensor<16x2x8x!tt.ptr<f32>>
// CHECK-NOT: tt.splat
func.func @broadcast_ptr_tensor(%arg0: tensor<2x8x!tt.ptr<f32>>) -> tensor<16x2x8x!tt.ptr<f32>> {
  %0 = tt.broadcast %arg0 : tensor<2x8x!tt.ptr<f32>> -> tensor<16x2x8x!tt.ptr<f32>>
  return %0 : tensor<16x2x8x!tt.ptr<f32>>
}

// -----

// CHECK-LABEL: @broadcast_tensor_2dto3d_nobroadcast
// CHECK-NEXT: %[[CST:.*]] = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<2x8xf32> -> tensor<1x2x8xf32>
// CHECK-NOT: tt.splat
// CHECK-NOT: tt.broadcast
func.func @broadcast_tensor_2dto3d_nobroadcast(%arg0: tensor<2x8xf32>) -> tensor<1x2x8xf32> {
  %0 = tt.broadcast %arg0 : tensor<2x8xf32> -> tensor<1x2x8xf32>
  return %0 : tensor<1x2x8xf32>
}
