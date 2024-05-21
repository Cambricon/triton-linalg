// RUN: triton-linalg-opt --canonicalize-triton %s -split-input-file | FileCheck %s

// CHECK-LABEL:   tt.func @masked_load_without_other_f32(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<8x!tt.ptr<f32>>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<8xi1>) -> tensor<8xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK:           %[[VAL_3:.*]] = tt.load %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<8x!tt.ptr<f32>>
tt.func @masked_load_without_other_f32(%arg0: tensor<8x!tt.ptr<f32>>, %arg1: tensor<8xi1>) -> tensor<8xf32> {
  %0 = tt.load %arg0, %arg1 : tensor<8x!tt.ptr<f32>>
  tt.return %0 : tensor<8xf32>
}

// -----
// CHECK-LABEL:   tt.func @masked_load_without_other_bf16(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<8x!tt.ptr<bf16>>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<8xi1>) -> tensor<8xbf16> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : tensor<8xbf16>
// CHECK:           %[[VAL_3:.*]] = tt.load %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<8x!tt.ptr<bf16>>
tt.func @masked_load_without_other_bf16(%arg0: tensor<8x!tt.ptr<bf16>>, %arg1: tensor<8xi1>) -> tensor<8xbf16> {
  %0 = tt.load %arg0, %arg1 : tensor<8x!tt.ptr<bf16>>
  tt.return %0 : tensor<8xbf16>
}

// -----
// CHECK-LABEL:   tt.func @masked_load_without_other_i32(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<8x!tt.ptr<i32>>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<8xi1>) -> tensor<8xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK:           %[[VAL_3:.*]] = tt.load %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<8x!tt.ptr<i32>>
tt.func @masked_load_without_other_i32(%arg0: tensor<8x!tt.ptr<i32>>, %arg1: tensor<8xi1>) -> tensor<8xi32> {
  %0 = tt.load %arg0, %arg1 : tensor<8x!tt.ptr<i32>>
  tt.return %0 : tensor<8xi32>
}

// -----
// CHECK-LABEL:   tt.func @scalar_masked_load_without_other_f32(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !tt.ptr<f32>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: i1) -> f32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = tt.load %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : !tt.ptr<f32>
tt.func @scalar_masked_load_without_other_f32(%arg0: !tt.ptr<f32>, %arg1: i1) -> f32 {
  %0 = tt.load %arg0, %arg1 : !tt.ptr<f32>
  tt.return %0 : f32
}

// -----
// CHECK-LABEL:   tt.func @blockptr_boundary_check_load_without_pad(
// CHECK-SAME:        %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i32) -> tensor<8xf32> {
// CHECK:           %[[VAL_5:.*]] = tt.make_tensor_ptr %[[VAL_0]], {{\[}}%[[VAL_2]]], {{\[}}%[[VAL_3]]], {{\[}}%[[VAL_4]]] {order = array<i32: 0>} : <tensor<8xf32>>
// CHECK:           %[[VAL_6:.*]] = tt.load %[[VAL_5]] {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8xf32>>
tt.func @blockptr_boundary_check_load_without_pad(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i64, %arg3: i64, %arg4: i32) -> tensor<8xf32> {
  %0 = tt.make_tensor_ptr %arg0, [%arg2], [%arg3], [%arg4] {order = array<i32: 0>} : <tensor<8xf32>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8xf32>>
  tt.return %1 : tensor<8xf32>
}
