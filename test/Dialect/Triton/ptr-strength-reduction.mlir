// RUN: triton-linalg-opt --ptr-strength-reduction -split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func @addptr_test_scalar
// CHECK:      %[[ARG4:.*]] = arith.addi %arg4, %arg3 : i32
// CHECK-NEXT: %[[ARG3:.*]] = arith.addi %[[ARG4]], %arg2 : i32
// CHECK-NEXT: %[[ARG2:.*]] = arith.addi %[[ARG3]], %c1_i32 : i32
// CHECK-NEXT: %[[ARG1:.*]] = arith.addi %[[ARG2]], %arg1 : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[ARG1]], %c42_i32 : i32
tt.func @addptr_test_scalar(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> f32 {
  %c1_i32 = arith.constant 1 : i32
  %c42_i32 = arith.constant 42 : i32
  %0 = tt.addptr %arg0, %c42_i32 : !tt.ptr<f32>, i32
  %1 = tt.addptr %0, %arg1 : !tt.ptr<f32>, i32
  %2 = tt.addptr %1, %c1_i32 : !tt.ptr<f32>, i32
  %3 = tt.addptr %2, %arg2 : !tt.ptr<f32>, i32
  %4 = tt.addptr %3, %arg3 : !tt.ptr<f32>, i32
  %5 = tt.addptr %4, %arg4 : !tt.ptr<f32>, i32
  %6 = tt.load %5 : !tt.ptr<f32>
  tt.return %6 : f32
}

// -----
// CHECK-LABEL: tt.func @addptr_test
// CHECK:      %[[TEMP5:.*]] = arith.addi %5, %4 : tensor<128xi32>
// CHECK-NEXT: %[[TEMP4:.*]] = arith.addi %[[TEMP5]], %3 : tensor<128xi32>
// CHECK-NEXT: %[[TEMP3:.*]] = arith.addi %[[TEMP4]], %2 : tensor<128xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = arith.addi %[[TEMP3]], %cst_2 : tensor<128xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %[[TEMP2]], %0 : tensor<128xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %1, %[[TEMP1]] : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
tt.func @addptr_test(%arg0: !tt.ptr<f32>) -> tensor<128xf32> {
  %cst = arith.constant dense<4> : tensor<128xi32>
  %cst_0 = arith.constant dense<3> : tensor<128xi32>
  %cst_1 = arith.constant dense<2> : tensor<128xi32>
  %cst_2 = arith.constant dense<42> : tensor<128xi32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.addptr %2, %cst_2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %4 = arith.divsi %0, %cst_1 : tensor<128xi32>
  %5 = tt.addptr %3, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %6 = arith.muli %0, %cst_1 : tensor<128xi32>
  %7 = tt.addptr %5, %6 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %8 = arith.muli %0, %cst_0 : tensor<128xi32>
  %9 = tt.addptr %7, %8 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %10 = arith.muli %0, %cst : tensor<128xi32>
  %11 = tt.addptr %9, %10 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %12 = tt.load %11 : tensor<128x!tt.ptr<f32>>
  tt.return %12 : tensor<128xf32>
}

// -----
// CHECK-LABEL: tt.func @expand_dims_test
// CHECK:      %[[TEMP0:.*]] = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = tt.expand_dims %[[TEMP0]] {axis = 0 : i32} : tensor<128x1xi32> -> tensor<1x128x1xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x128x1x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP2]], %[[TEMP1]] : tensor<1x128x1x!tt.ptr<f32>>, tensor<1x128x1xi32>
tt.func @expand_dims_test(%arg0: !tt.ptr<f32>) -> tensor<1x128x1xf32> {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128x!tt.ptr<f32>> -> tensor<128x1x!tt.ptr<f32>>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128x1x!tt.ptr<f32>> -> tensor<1x128x1x!tt.ptr<f32>>
  %5 = tt.load %4 : tensor<1x128x1x!tt.ptr<f32>>
  tt.return %5 : tensor<1x128x1xf32>
}

// -----
// CHECK-LABEL: tt.func @broadcast_test
// CHECK:      %[[TEMP0:.*]] = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.muli %[[TEMP0]], %cst : tensor<128x1xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.broadcast %[[TEMP1]] : tensor<128x1xi32> -> tensor<128x64xi32>
// CHECK-NEXT: %[[TEMP4:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP5:.*]] = tt.broadcast %[[TEMP2]] : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT: %[[TEMP6:.*]] = arith.addi %[[TEMP5]], %[[TEMP3]] : tensor<128x64xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP4]], %[[TEMP6]] : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
tt.func @broadcast_test(%arg0: !tt.ptr<f32>) -> tensor<128x64xf32> {
  %cst = arith.constant dense<42> : tensor<128x1xi32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %3 = arith.muli %2, %cst : tensor<128x1xi32>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  %5 = tt.addptr %4, %3 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %7 = tt.broadcast %5 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x64x!tt.ptr<f32>>
  %8 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<128x64xi32>
  %9 = tt.addptr %7, %8 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %10 = tt.load %9 : tensor<128x64x!tt.ptr<f32>>
  tt.return %10 : tensor<128x64xf32>
}

// -----
// CHECK-LABEL: tt.func @view_test
// CHECK:      %[[RANGE:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = tt.expand_dims %[[RANGE]] {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.reshape %[[TEMP1]] {allow_reorder = false} : tensor<128x1xi32> -> tensor<32x4xi32>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP3]], %[[TEMP2]] : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
tt.func @view_test(%arg0: !tt.ptr<f32>) -> tensor<32x4xf32> {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  %3 = tt.addptr %2, %1 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  %4 = tt.reshape %3 {allow_reorder = false} : tensor<128x1x!tt.ptr<f32>> -> tensor<32x4x!tt.ptr<f32>>
  %5 = tt.load %4 : tensor<32x4x!tt.ptr<f32>>
  tt.return %5 : tensor<32x4xf32>
}

// -----
// CHECK-LABEL: tt.func @trans_test
// CHECK:      %[[TEMP0:.*]] = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.broadcast %[[TEMP0]] : tensor<128x1xi32> -> tensor<128x64xi32>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.broadcast %[[TEMP1]] : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT: %[[TEMP4:.*]] = arith.addi %[[TEMP2]], %[[TEMP3]] : tensor<128x64xi32>
// CHECK-NEXT: %[[TEMP5:.*]] = tt.trans %[[TEMP4]] {order = array<i32: 1, 0>} : tensor<128x64xi32> -> tensor<64x128xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>>
tt.func @trans_test(%arg0: !tt.ptr<f32>) -> tensor<64x128xf32> {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %4 = tt.broadcast %2 : tensor<128x1xi32> -> tensor<128x64xi32>
  %5 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<128x64xi32>
  %6 = arith.addi %4, %5 : tensor<128x64xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %6 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<128x64x!tt.ptr<f32>> -> tensor<64x128x!tt.ptr<f32>>
  %10 = tt.load %9 : tensor<64x128x!tt.ptr<f32>>
  tt.return %10 : tensor<64x128xf32>
}

// -----
// CHECK-LABEL: tt.func @bitcast_test
// CHECK:      %[[RANGE:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.splat %[[TEMP1]] : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.load %[[TEMP2]] : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP4:.*]] = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK-NEXT: %[[TEMP5:.*]] = tt.splat %[[TEMP4]] : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP6:.*]] = tt.addptr %[[TEMP5]], %[[RANGE]] : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.load %[[TEMP6]] : tensor<64x!tt.ptr<f32>>
tt.func @bitcast_test(%arg0: !tt.ptr<f16>) -> (tensor<64xf32>, tensor<64xf32>) {
  %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  %2 = tt.bitcast %1 : tensor<64x!tt.ptr<f16>> -> tensor<64x!tt.ptr<f32>>
  %3 = tt.load %2 : tensor<64x!tt.ptr<f32>>
  %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  %5 = tt.addptr %4, %0 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  %6 = tt.bitcast %5 : tensor<64x!tt.ptr<f16>> -> tensor<64x!tt.ptr<f32>>
  %7 = tt.load %6 : tensor<64x!tt.ptr<f32>>
  tt.return %3, %7 : tensor<64xf32>, tensor<64xf32>
}

// -----
// CHECK-LABEL: tt.func @advance_test
// CHECK:      %[[TEMP0:.*]] = arith.addi %arg3, %arg1 : i32
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %arg4, %arg2 : i32
// CHECK-NEXT: %[[RESULT:.*]] = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%[[TEMP0]], %[[TEMP1]]] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @advance_test(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> tensor<32x32xf32> {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %1 = tt.advance %0, [%arg1, %arg2] : <tensor<32x32xf32>>
  %2 = tt.advance %1, [%arg3, %arg4] : <tensor<32x32xf32>>
  %3 = tt.load %2 : !tt.ptr<tensor<32x32xf32>>
  tt.return %3 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @control_flow_br_test
// CHECK:      %[[TEMP0:.*]] = arith.addi %0, %cst : tensor<128xi32>
// CHECK-NEXT: cf.br ^bb1(%arg0, %0, %arg0, %[[TEMP0]] : !tt.ptr<f32>, tensor<128xi32>, !tt.ptr<f32>, tensor<128xi32>)
// CHECK:    ^bb1(%2: !tt.ptr<f32>, %3: tensor<128xi32>, %4: !tt.ptr<f32>, %5: tensor<128xi32>):  // pred: ^bb0
// CHECK-NEXT: %[[TEMP1:.*]] = tt.splat %4 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.addptr %[[TEMP1]], %5 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.splat %2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP4:.*]] = tt.addptr %[[TEMP3]], %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT: %[[TEMP5:.*]] = tt.load %[[TEMP4]] : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP6:.*]] = tt.load %[[TEMP2]] : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: tt.return %[[TEMP5]], %[[TEMP6]] : tensor<128xf32>, tensor<128xf32>
tt.func @control_flow_br_test(%arg0: !tt.ptr<f32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %cst = arith.constant dense<42> : tensor<128xi32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.addptr %2, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  cf.br ^bb1(%2, %3 : tensor<128x!tt.ptr<f32>>, tensor<128x!tt.ptr<f32>>)
^bb1(%4: tensor<128x!tt.ptr<f32>>, %5: tensor<128x!tt.ptr<f32>>):  // pred: ^bb0
  %6 = tt.load %4 : tensor<128x!tt.ptr<f32>>
  %7 = tt.load %5 : tensor<128x!tt.ptr<f32>>
  tt.return %6, %7 : tensor<128xf32>, tensor<128xf32>
}

// -----
// CHECK-LABEL: tt.func @control_flow_br_with_blockptr_test
// CHECK: cf.br ^bb1(%arg0, %c128_i64, %c64_i64, %c1_i64, %c1_i64, %c0_i32, %c0_i32 : !tt.ptr<f32>, i64, i64, i64, i64, i32, i32)
// CHECK-NEXT: ^bb1(%0: !tt.ptr<f32>, %1: i64, %2: i64, %3: i64, %4: i64, %5: i32, %6: i32):  // pred: ^bb0
// CHECK-NEXT:   %[[ARG1:.*]] = arith.addi %arg1, %5 : i32
// CHECK-NEXT:   %[[ARG2:.*]] = arith.addi %arg2, %6 : i32
// CHECK-NEXT:   %[[TENSOR_PTR:.*]] = tt.make_tensor_ptr %0, [%1, %2], [%3, %4], [%7, %8] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @control_flow_br_with_blockptr_test(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<32x32xf32> {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  cf.br ^bb1(%0 : !tt.ptr<tensor<32x32xf32>>)
^bb1(%1: !tt.ptr<tensor<32x32xf32>>):  // pred: ^bb0
  %2 = tt.advance %1, [%arg1, %arg2] : <tensor<32x32xf32>>
  %3 = tt.load %2 : !tt.ptr<tensor<32x32xf32>>
  tt.return %3 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @control_flow_cond_br_test
// CHECK:      cf.cond_br %arg1, ^bb1(%arg0, %0 : !tt.ptr<f32>, tensor<128xi32>), ^bb2(%arg0, %0 : !tt.ptr<f32>, tensor<128xi32>)
// CHECK:    ^bb1(%1: !tt.ptr<f32>, %2: tensor<128xi32>):  // pred: ^bb0
// CHECK-NEXT: %[[TEMP0:.*]] = arith.addi %2, %cst : tensor<128xi32>
// CHECK-NEXT: cf.br ^bb2(%1, %[[TEMP0]] : !tt.ptr<f32>, tensor<128xi32>)
// CHECK:    ^bb2(%4: !tt.ptr<f32>, %5: tensor<128xi32>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT: %[[TEMP1:.*]] = tt.splat %4 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP2:.*]] = tt.addptr %[[TEMP1]], %5 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.load %[[TEMP2]] : tensor<128x!tt.ptr<f32>>
tt.func @control_flow_cond_br_test(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<128xf32> {
  %cst = arith.constant dense<42> : tensor<128xi32>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  cf.cond_br %arg1, ^bb1(%2 : tensor<128x!tt.ptr<f32>>), ^bb2(%2 : tensor<128x!tt.ptr<f32>>)
^bb1(%3: tensor<128x!tt.ptr<f32>>):  // pred: ^bb0
  %4 = tt.addptr %3, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  cf.br ^bb2(%4 : tensor<128x!tt.ptr<f32>>)
^bb2(%5: tensor<128x!tt.ptr<f32>>):  // 2 preds: ^bb0, ^bb1
  %6 = tt.load %5 : tensor<128x!tt.ptr<f32>>
  tt.return %6 : tensor<128xf32>
}

// -----
// CHECK-LABEL: tt.func @for_test
// CHECK: %[[RESULT:.*]]:[[RESULT_NUL:.*]] = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %arg0, %arg3 = %0) -> (!tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP0:.*]] = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:   %[[TEMP1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP2:.*]] = tt.splat %[[TEMP0]] : i32 -> tensor<32xi32>
// CHECK-NEXT:   %[[TEMP3:.*]] = arith.addi %[[TEMP1]], %[[TEMP2]] : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP4:.*]] = arith.addi %[[TEMP3]], %arg3 : tensor<32xi32>
// CHECK-NEXT:   scf.yield %arg2, %[[TEMP4]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP5:.*]] = tt.splat %[[RESULT]]#0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT1:.*]] = tt.addptr %[[TEMP5]], %[[RESULT]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @for_test(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %2) -> (tensor<32x!tt.ptr<f32>>) {
    %5 = arith.index_cast %arg1 : index to i32
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %7 = tt.splat %5 : i32 -> tensor<32xi32>
    %8 = arith.addi %6, %7 : tensor<32xi32>
    %9 = tt.addptr %arg2, %8 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %9 : tensor<32x!tt.ptr<f32>>
  }
  %4 = tt.load %3 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
tt.func @for_with_execute_region_test(%arg0 : i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %arg1 = %c0 to %c10 step %c1 {
    scf.execute_region {
      cf.cond_br %arg0, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      scf.yield
    ^bb2:  // pred: ^bb0
      scf.yield
    }
  }
  tt.return
}
// CHECK-LABEL:   tt.func @for_with_execute_region_test(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i1) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK:             scf.execute_region {
// CHECK:               cf.cond_br %[[VAL_0]], ^bb1, ^bb1
// CHECK:             ^bb1:
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }

// -----
// CHECK-LABEL: tt.func @for_with_blockptr_test
// CHECK:      %[[ARG0:.*]]:14 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %arg0, %arg4 = %c128_i64, %arg5 = %c64_i64, %arg6 = %c1_i64, %arg7 = %c1_i64, %arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %arg0, %arg11 = %c128_i64, %arg12 = %c64_i64, %arg13 = %c1_i64, %arg14 = %c1_i64, %arg15 = %c0_i32, %arg16 = %c0_i32) -> (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32, !tt.ptr<f32>, i64, i64, i64, i64, i32, i32) {
// CHECK-NEXT:   %[[ARG2:.*]] = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:   %[[ARG6:.*]] = arith.addi %arg1, %arg8 : i32
// CHECK-NEXT:   %[[ARG7:.*]] = arith.addi %[[ARG2]], %arg9 : i32
// CHECK-NEXT:   %[[ARG8:.*]] = arith.addi %[[ARG2]], %arg15 : i32
// CHECK-NEXT:   %[[ARG9:.*]] = arith.addi %[[ARG2]], %arg16 : i32
// CHECK-NEXT:   scf.yield %arg3, %arg4, %arg5, %arg6, %arg7, %6, %7, %arg10, %arg11, %arg12, %arg13, %arg14, %[[ARG8]], %[[ARG9]] : !tt.ptr<f32>, i64, i64, i64, i64, i32, i32, !tt.ptr<f32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = tt.make_tensor_ptr %[[ARG0]]#7, [%[[ARG0]]#8, %[[ARG0]]#9], [%[[ARG0]]#10, %[[ARG0]]#11], [%[[ARG0]]#12, %[[ARG0]]#13] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
// CHECK-NEXT:   %2 = tt.make_tensor_ptr %[[ARG0]]#0, [%[[ARG0]]#1, %[[ARG0]]#2], [%[[ARG0]]#3, %[[ARG0]]#4], [%[[ARG0]]#5, %[[ARG0]]#6] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @for_with_blockptr_test(%arg0: !tt.ptr<f32>, %arg1: i32) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %1 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %2:2 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %0, %arg4 = %1) -> (!tt.ptr<tensor<32x32xf32>>, !tt.ptr<tensor<32x32xf32>>) {
    %5 = arith.index_cast %arg2 : index to i32
    %6 = tt.advance %arg3, [%arg1, %5] : <tensor<32x32xf32>>
    %7 = tt.advance %arg4, [%5, %5] : <tensor<32x32xf32>>
    scf.yield %6, %7 : !tt.ptr<tensor<32x32xf32>>, !tt.ptr<tensor<32x32xf32>>
  }
  %3 = tt.load %2#0 : !tt.ptr<tensor<32x32xf32>>
  %4 = tt.load %2#1 : !tt.ptr<tensor<32x32xf32>>
  tt.return %3, %4 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @nest_for_test
// CHECK:      %[[RESULT1:.*]]:3 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %0, %arg3 = %arg0, %arg4 = %0) -> (tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP0:.*]] = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:   %[[TEMP1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP2:.*]] = tt.splat %[[TEMP0]] : i32 -> tensor<32xi32>
// CHECK-NEXT:   %[[RESULT2:.*]]:3 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %7, %arg7 = %arg3, %arg8 = %arg4) -> (tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:     %[[TEMP3:.*]] = arith.index_cast %arg5 : index to i32
// CHECK-NEXT:     %[[TEMP4:.*]] = tt.splat %[[TEMP3]] : i32 -> tensor<32xi32>
// CHECK-NEXT:     %[[TEMP5:.*]] = arith.addi %[[TEMP4]], %arg8 : tensor<32xi32>
// CHECK-NEXT:     scf.yield %[[TEMP4]], %arg7, %[[TEMP5]] : tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[TEMP6:.*]] = arith.addi %[[TEMP1]], %[[RESULT2]]#0 : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP7:.*]] = arith.addi %[[TEMP6]], %[[RESULT2]]#2 : tensor<32xi32>
// CHECK-NEXT:   scf.yield %[[TEMP6]], %8#1, %[[TEMP7]] : tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP8:.*]] = tt.splat %[[RESULT1]]#1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %3 = tt.addptr %[[TEMP8]], %[[RESULT1]]#2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @nest_for_test(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3:2 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %0, %arg3 = %2) -> (tensor<32xi32>, tensor<32x!tt.ptr<f32>>) {
    %5 = arith.index_cast %arg1 : index to i32
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %7 = tt.splat %5 : i32 -> tensor<32xi32>
    %8:2 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %7, %arg6 = %arg3) -> (tensor<32xi32>, tensor<32x!tt.ptr<f32>>) {
      %11 = arith.index_cast %arg4 : index to i32
      %12 = tt.splat %11 : i32 -> tensor<32xi32>
      %13 = tt.addptr %arg6, %12 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      scf.yield %12, %13 : tensor<32xi32>, tensor<32x!tt.ptr<f32>>
    }
    %9 = arith.addi %6, %8#0 : tensor<32xi32>
    %10 = tt.addptr %8#1, %9 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %9, %10 : tensor<32xi32>, tensor<32x!tt.ptr<f32>>
  }
  %4 = tt.load %3#1 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @for_multi_ptr_test
// CHECK:      %[[RESULT1:.*]]{{:.*}} = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %0, %arg4 = %arg0, %arg5 = %0, %arg6 = %arg1, %arg7 = %0) -> (tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP0:.*]] = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:   %[[TEMP1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP2:.*]] = tt.splat %[[TEMP0]] : i32 -> tensor<32xi32>
// CHECK-NEXT:   %[[TEMP3:.*]] = arith.addi %[[TEMP1]], %[[TEMP2]] : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP4:.*]] = arith.addi %[[TEMP3]], %arg5 : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP5:.*]] = arith.addi %[[TEMP2]], %arg7 : tensor<32xi32>
// CHECK-NEXT:   scf.yield %[[TEMP3]], %arg4, %[[TEMP4]], %arg6, %[[TEMP5]] : tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP6:.*]] = tt.splat %[[RESULT1]]#3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT2:.*]] = tt.addptr %[[TEMP6]], %[[RESULT1]]#4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT: %[[TEMP7:.*]] = tt.splat %[[RESULT1]]#1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT3:.*]] = tt.addptr %[[TEMP7]], %[[RESULT1]]#2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @for_multi_ptr_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) -> (tensor<32xf32>, tensor<32xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %4 = tt.addptr %3, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %5:3 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %0, %arg4 = %2, %arg5 = %4) -> (tensor<32xi32>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>) {
    %8 = arith.index_cast %arg2 : index to i32
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %10 = tt.splat %8 : i32 -> tensor<32xi32>
    %11 = arith.addi %9, %10 : tensor<32xi32>
    %12 = tt.addptr %arg4, %11 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %13 = tt.addptr %arg5, %10 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %11, %12, %13 : tensor<32xi32>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>
  }
  %6 = tt.load %5#1 : tensor<32x!tt.ptr<f32>>
  %7 = tt.load %5#2 : tensor<32x!tt.ptr<f32>>
  tt.return %6, %7 : tensor<32xf32>, tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @while_test
// CHECK:        %[[RESULT1:.*]]:2 = scf.while (%arg2 = %arg0, %arg3 = %0) : (!tt.ptr<f32>, tensor<32xi32>) -> (!tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP0:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP1:.*]] = arith.addi %[[TEMP0]], %arg3 : tensor<32xi32>
// CHECK-NEXT:  scf.condition(%arg1) %arg2, %[[TEMP1]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: } do {
// CHECK-NEXT:   ^bb0(%arg2: !tt.ptr<f32>, %arg3: tensor<32xi32>):
// CHECK-NEXT:     %[[TEMP2:.*]] = arith.addi %arg3, %cst
// CHECK-NEXT:     scf.yield %arg2, %[[TEMP2]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP3:.*]] = tt.splat %[[RESULT1]]#0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT2:.*]] = tt.addptr %[[TEMP3]], %[[RESULT1]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @while_test(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<32xf32> {
  %cst = arith.constant dense<2> : tensor<32xi32>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = scf.while (%arg2 = %2) : (tensor<32x!tt.ptr<f32>>) -> tensor<32x!tt.ptr<f32>> {
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.addptr %arg2, %5 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.condition(%arg1) %6 : tensor<32x!tt.ptr<f32>>
  } do {
  ^bb0(%arg2: tensor<32x!tt.ptr<f32>>):
    %5 = tt.addptr %arg2, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %5 : tensor<32x!tt.ptr<f32>>
  }
  %4 = tt.load %3 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @while_with_block_ptr_test
// CHECK:        %[[ARG0:.*]]:7 = scf.while (%arg4 = %arg0, %arg5 = %c128_i64, %arg6 = %c64_i64, %arg7 = %c1_i64, %arg8 = %c1_i64, %arg9 = %c0_i32, %arg10 = %c0_i32) : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) {
// CHECK-NEXT:   %[[COND_ARG4:.*]] = arith.addi %arg2, %arg9 : i32
// CHECK-NEXT:   %[[COND_ARG5:.*]] = arith.addi %arg3, %arg10 : i32
// CHECK-NEXT:   scf.condition(%arg1) %arg4, %arg5, %arg6, %arg7, %arg8, %[[COND_ARG4]], %[[COND_ARG5]] : !tt.ptr<f32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: } do {
// CHECK-NEXT:  ^bb0(%arg4: !tt.ptr<f32>, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i32, %arg10: i32):
// CHECK-NEXT:   %[[BODY_ARG4:.*]] = arith.addi %arg2, %arg9 : i32
// CHECK-NEXT:   %[[BODY_ARG5:.*]] = arith.addi %arg3, %arg10 : i32
// CHECK-NEXT:   scf.yield %arg4, %arg5, %arg6, %arg7, %arg8, %[[BODY_ARG4]], %[[BODY_ARG5]] : !tt.ptr<f32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: }
// CHECK-NEXT: tt.make_tensor_ptr %[[ARG0]]#0, [%[[ARG0]]#1, %[[ARG0]]#2], [%[[ARG0]]#3, %[[ARG0]]#4], [%[[ARG0]]#5, %[[ARG0]]#6] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @while_with_block_ptr_test(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: i32, %arg3: i32) -> tensor<32x32xf32> {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %1 = scf.while (%arg4 = %0) : (!tt.ptr<tensor<32x32xf32>>) -> !tt.ptr<tensor<32x32xf32>> {
    %3 = tt.advance %arg4, [%arg2, %arg3] : <tensor<32x32xf32>>
    scf.condition(%arg1) %3 : !tt.ptr<tensor<32x32xf32>>
  } do {
  ^bb0(%arg4: !tt.ptr<tensor<32x32xf32>>):
    %3 = tt.advance %arg4, [%arg2, %arg3] : <tensor<32x32xf32>>
    scf.yield %3 : !tt.ptr<tensor<32x32xf32>>
  }
  %2 = tt.load %1 : !tt.ptr<tensor<32x32xf32>>
  tt.return %2 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @if_test
// CHECK:      %[[RESULT1:.*]] = scf.if %arg1 -> (tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP0:.*]] = arith.addi %0, %cst : tensor<32xi32>
// CHECK-NEXT:   scf.yield %[[TEMP0]] : tensor<32xi32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[TEMP1:.*]] = arith.addi %0, %cst_0 : tensor<32xi32>
// CHECK-NEXT:   scf.yield %[[TEMP1]] : tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP2:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT2:.*]] = tt.addptr %[[TEMP2]], %[[RESULT1]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @if_test(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<32xf32> {
  %cst = arith.constant dense<42> : tensor<32xi32>
  %cst_0 = arith.constant dense<2> : tensor<32xi32>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = scf.if %arg1 -> (tensor<32x!tt.ptr<f32>>) {
    %5 = tt.addptr %2, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %5 : tensor<32x!tt.ptr<f32>>
  } else {
    %5 = tt.addptr %2, %cst_0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %5 : tensor<32x!tt.ptr<f32>>
  }
  %4 = tt.load %3 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @if_with_blockptr_test
// CHECK:      %[[ARG0:.*]]:7 = scf.if %arg1 -> (tensor<32x32xf32>, i64, i64, i64, i64, i32, i32) {
// CHECK-NEXT:   %[[COND_PTR:.*]] = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%arg2, %arg3] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
// CHECK-NEXT:   %[[COND_LOAD:.*]] = tt.load %[[COND_PTR]] : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:   scf.yield %[[COND_LOAD]], %c128_i64, %c64_i64, %c1_i64, %c1_i64, %arg2, %arg3 : tensor<32x32xf32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[ELSE_PTR:.*]] = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%arg2, %arg3] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
// CHECK-NEXT:   %[[ELSE_LOAD:.*]] = tt.load %[[ELSE_PTR]] : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:   scf.yield %[[ELSE_LOAD]], %c128_i64, %c64_i64, %c1_i64, %c1_i64, %arg2, %arg3 : tensor<32x32xf32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: }
// CHECK-NEXT: tt.make_tensor_ptr %arg0, [%[[ARG0]]#1, %[[ARG0]]#2], [%[[ARG0]]#3, %[[ARG0]]#4], [%[[ARG0]]#5, %[[ARG0]]#6] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @if_with_blockptr_test(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: i32, %arg3: i32) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %1:2 = scf.if %arg1 -> (!tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>) {
    %3 = tt.advance %0, [%arg2, %arg3] : <tensor<32x32xf32>>
    %4 = tt.load %3 : !tt.ptr<tensor<32x32xf32>>
    scf.yield %3, %4 : !tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>
  } else {
    %3 = tt.advance %0, [%arg2, %arg3] : <tensor<32x32xf32>>
    %4 = tt.load %3 : !tt.ptr<tensor<32x32xf32>>
    scf.yield %3, %4 : !tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>
  }
  %2 = tt.load %1#0 : !tt.ptr<tensor<32x32xf32>>
  tt.return %2, %1#1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @int_to_ptr_scalar
// CHECK:      %[[TEMP0:.*]] = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %arg1, %arg1 : i32
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP0]], %[[TEMP1]] : !tt.ptr<f32>, i32
tt.func @int_to_ptr_scalar(%arg0: i64, %arg1: i32) -> !tt.ptr<f32> {
  %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
  %1 = tt.addptr %0, %arg1 : !tt.ptr<f32>, i32
  %2 = tt.addptr %1, %arg1 : !tt.ptr<f32>, i32
  tt.return %2 : !tt.ptr<f32>
}

// -----
// CHECK-LABEL: tt.func @int_to_ptr_tensor
// CHECK:      %[[TEMP0:.*]] = tt.int_to_ptr %arg0 : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %arg1, %arg1 : tensor<32xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP0]], %[[TEMP1]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @int_to_ptr_tensor(%arg0: tensor<32xi64>, %arg1: tensor<32xi32>) -> tensor<32x!tt.ptr<f32>> {
  %0 = tt.int_to_ptr %arg0 : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %2 = tt.addptr %1, %arg1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  tt.return %2 : tensor<32x!tt.ptr<f32>>
}

// -----
// CHECK-LABEL: tt.func @ptr_to_int_scalar
// CHECK:      %[[TEMP0:.*]] = arith.addi %0, %arg1 : i64
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %[[TEMP0]], %arg1 : i64
// CHECK-NEXT: %[[TEMP2:.*]] = tt.int_to_ptr %[[TEMP1]] : i64 -> !tt.ptr<f32>
// CHECK-NEXT: %[[TEMP3:.*]] = arith.addi %arg2, %arg2 : i32
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP2]], %[[TEMP3]] : !tt.ptr<f32>, i32
tt.func @ptr_to_int_scalar(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i32) -> !tt.ptr<f32> {
  %0 = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
  %1 = arith.addi %0, %arg1 : i64
  %2 = arith.addi %1, %arg1 : i64
  %3 = tt.int_to_ptr %2 : i64 -> !tt.ptr<f32>
  %4 = tt.addptr %3, %arg2 : !tt.ptr<f32>, i32
  %5 = tt.addptr %4, %arg2 : !tt.ptr<f32>, i32
  tt.return %5 : !tt.ptr<f32>
}

// -----
// CHECK-LABEL: tt.func @ptr_to_int_tensor
// CHECK:      %[[TEMP0:.*]] = tt.ptr_to_int %0 : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %[[TEMP0]], %arg1 : tensor<32xi64>
// CHECK-NEXT: %[[TEMP2:.*]] = arith.addi %[[TEMP1]], %arg1 : tensor<32xi64>
// CHECK-NEXT: %[[TEMP3:.*]] = tt.int_to_ptr %[[TEMP2]] : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP4:.*]] = arith.addi %arg2, %arg2 : tensor<32xi32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP3]], %[[TEMP4]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @ptr_to_int_tensor(%arg0: !tt.ptr<f32>, %arg1: tensor<32xi64>, %arg2: tensor<32xi32>) -> tensor<32x!tt.ptr<f32>> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %1 = tt.ptr_to_int %0 : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
  %2 = arith.addi %1, %arg1 : tensor<32xi64>
  %3 = arith.addi %2, %arg1 : tensor<32xi64>
  %4 = tt.int_to_ptr %3 : tensor<32xi64> -> tensor<32x!tt.ptr<f32>>
  %5 = tt.addptr %4, %arg2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %6 = tt.addptr %5, %arg2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  tt.return %6 : tensor<32x!tt.ptr<f32>>
}

// -----
// CHECK-LABEL: tt.func @if_multi_return_test
// CHECK:      %[[TEMP0:.*]]:2 = scf.if %arg1 -> (tensor<32xi64>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP1:.*]] = arith.addi %0, %cst : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP2:.*]] = tt.addptr %1, %[[TEMP1]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:   %[[TEMP3:.*]] = tt.ptr_to_int %[[TEMP2]] : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
// CHECK-NEXT:   scf.yield %[[TEMP3]], %[[TEMP1]] : tensor<32xi64>, tensor<32xi32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[TEMP1:.*]] = arith.addi %0, %cst_0 : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP2:.*]] = tt.addptr %1, %[[TEMP1]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:   %[[TEMP3:.*]] = tt.ptr_to_int %[[TEMP2]] : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
// CHECK-NEXT:   scf.yield %[[TEMP3]], %[[TEMP1]] : tensor<32xi64>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP4:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %[[TEMP4]], %[[TEMP0]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @if_multi_return_test(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<32xf32> {
  %cst = arith.constant dense<42> : tensor<32xi32>
  %cst_0 = arith.constant dense<2> : tensor<32xi32>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3:2 = scf.if %arg1 -> (tensor<32xi64>, tensor<32x!tt.ptr<f32>>) {
    %5 = tt.addptr %2, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %6 = tt.ptr_to_int %5 : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
    scf.yield %6, %5 : tensor<32xi64>, tensor<32x!tt.ptr<f32>>
  } else {
    %5 = tt.addptr %2, %cst_0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %6 = tt.ptr_to_int %5 : tensor<32x!tt.ptr<f32>> -> tensor<32xi64>
    scf.yield %6, %5 : tensor<32xi64>, tensor<32x!tt.ptr<f32>>
  }
  tt.print "%9#0" {hex = false} : %3#0 : tensor<32xi64>
  %4 = tt.load %3#1 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @for_multi_iter_args_test
// CHECK: %[[RESULT:.*]] = arith.addi %54, %0 : tensor<32xi32>
// CHECK: %[[RESULTS:.*]]:22 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %arg0, %arg3 = %0, %arg4 = %arg0, %arg5 = %1, %arg6 = %arg0, %arg7 = %3, %arg8 = %arg0, %arg9 = %6, %arg10 = %arg0, %arg11 = %10, %arg12 = %arg0, %arg13 = %15, %arg14 = %arg0, %arg15 = %21, %arg16 = %arg0, %arg17 = %28, %arg18 = %arg0, %arg19 = %36, %arg20 = %arg0, %arg21 = %45, %arg22 = %arg0, %arg23 = %55) -> (!tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>) {
// CHECK: scf.yield %arg2, %64, %arg4, %65, %arg6, %66, %arg8, %67, %arg10, %68, %arg12, %69, %arg14, %70, %arg16, %71, %arg18, %72, %arg20, %73, %arg22, %74 : !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[RESULT2:.*]] = tt.splat %[[RESULTS]]#20 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT  %[[RESULT3:.*]] = tt.addptr %[[RESULT2]], %[[RESULTS]]#21 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @for_multi_iter_args_test(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = tt.addptr %2, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %4 = tt.addptr %3, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %6 = tt.addptr %5, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %7 = tt.addptr %6, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %8 = tt.addptr %7, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %9 = tt.addptr %8, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %10 = tt.addptr %9, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %11 = tt.addptr %10, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %12 = tt.addptr %11, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %13:11 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %2, %arg3 = %3, %arg4 = %4, %arg5 = %5, %arg6 = %6, %arg7 = %7, %arg8 = %8, %arg9 = %9, %arg10 = %10, %arg11 = %11, %arg12 = %12) -> (tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>) {
    %15 = arith.index_cast %arg1 : index to i32
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %17 = tt.splat %15 : i32 -> tensor<32xi32>
    %18 = arith.addi %16, %17 : tensor<32xi32>
    %19 = tt.addptr %arg2, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %20 = tt.addptr %arg3, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %21 = tt.addptr %arg4, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %22 = tt.addptr %arg5, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %23 = tt.addptr %arg6, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %24 = tt.addptr %arg7, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %25 = tt.addptr %arg8, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %26 = tt.addptr %arg9, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %27 = tt.addptr %arg10, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %28 = tt.addptr %arg11, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %29 = tt.addptr %arg12, %18 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29 : tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>, tensor<32x!tt.ptr<f32>>
  }
  %14 = tt.load %13#10 : tensor<32x!tt.ptr<f32>>
  tt.return %14 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @scalar_with_offset_producer_test
// CHECK: %[[RESULT1:.*]] = arith.addi %arg1, %arg1 : i32
// CHECK: %[[RESULT2:.*]] = tt.addptr %0, %[[RESULT1]] : !tt.ptr<f16>, i32
// CHECK: %[[RESULT3:.*]] = tt.bitcast %[[RESULT2]] : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK: %[[RESULT4:.*]] = tt.splat %[[RESULT3]] : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT5:.*]] = tt.splat %[[RESULT3]] : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT6:.*]] = tt.load %[[RESULT5]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT7:.*]] = tt.load %[[RESULT4]] : tensor<1x64x!tt.ptr<f32>>
tt.func @scalar_with_offset_producer_test(%arg0: i64, %arg1: i32) -> tensor<1x64xf32> {
  %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f16>
  %1 = tt.addptr %0, %arg1 : !tt.ptr<f16>, i32
  // COM: %1 is a scalar ptr with offset.
  %2 = tt.addptr %1, %arg1 : !tt.ptr<f16>, i32
  %3 = tt.bitcast %2 : !tt.ptr<f16> -> !tt.ptr<f32>
  %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %5 = tt.splat %3 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  // COM: End of test.
  %6 = tt.load %5 : tensor<1x64x!tt.ptr<f32>>
  %7 = tt.load %4 : tensor<1x64x!tt.ptr<f32>>
  %8 = arith.addf %6, %7 : tensor<1x64xf32>
  tt.return %8 : tensor<1x64xf32>
}

// -----
// CHECK-LABEL: tt.func @scalar_without_offset_producer_test
// CHECK: %[[RESULT1:.*]] = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK: %[[RESULT2:.*]] = tt.splat %[[RESULT1]] : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT3:.*]] = tt.load %[[RESULT2]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT4:.*]] = tt.load %[[RESULT2]] : tensor<1x64x!tt.ptr<f32>>
tt.func @scalar_without_offset_producer_test(%arg0: !tt.ptr<f16>, %arg1: i32) -> tensor<1x64xf32> {
  // COM: %arg0 is a scalar ptr without offset.
  %0 = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
  %1 = tt.splat %0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  // COM: End of test.
  %2 = tt.load %1 : tensor<1x64x!tt.ptr<f32>>
  %3 = tt.load %1 : tensor<1x64x!tt.ptr<f32>>
  %4 = arith.addf %2, %3 : tensor<1x64xf32>
  tt.return %4 : tensor<1x64xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_with_offset_producer_test
// CHECK: %[[RESULT1:.*]] = arith.addi %arg2, %arg1 : tensor<64xi32>
// CHECK: %[[RESULT2:.*]] = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK: %[[RESULT3:.*]] = tt.expand_dims %[[RESULT1]] {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK: %[[RESULT4:.*]] = tt.trans %[[RESULT3]] {order = array<i32: 1, 0>} : tensor<1x64xi32> -> tensor<64x1xi32>
// CHECK: %[[RESULT5:.*]] = tt.splat %[[RESULT2]] : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
// CHECK: %[[RESULT6:.*]] = tt.addptr %[[RESULT5]], %[[RESULT4]] : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
tt.func @tensor_with_offset_producer_test(%arg0: !tt.ptr<f16>, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>) -> tensor<64x1xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  %1 = tt.addptr %0, %arg1 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  // COM: %1 is a tensor of ptr with offset.
  %2 = tt.addptr %1, %arg2 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  %3 = tt.bitcast %2 : tensor<64x!tt.ptr<f16>> -> tensor<64x!tt.ptr<f32>>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64x!tt.ptr<f32>> -> tensor<1x64x!tt.ptr<f32>>
  %5 = tt.trans %4 {order = array<i32: 1, 0>} : tensor<1x64x!tt.ptr<f32>> -> tensor<64x1x!tt.ptr<f32>>
  // COM: End of test.
  %6 = tt.load %5 : tensor<64x1x!tt.ptr<f32>>
  tt.return %6 : tensor<64x1xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_without_offset_producer_test
// CHECK: %[[RESULT1:.*]] = tt.bitcast %arg0 : !tt.ptr<f16> -> !tt.ptr<f32>
// CHECK: %[[RESULT2:.*]] = tt.expand_dims %arg2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK: %[[RESULT3:.*]] = tt.trans %[[RESULT2]] {order = array<i32: 1, 0>} : tensor<1x64xi32> -> tensor<64x1xi32>
// CHECK: %[[RESULT4:.*]] = tt.splat %[[RESULT1]] : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
// CHECK: %[[RESULT5:.*]] = tt.addptr %[[RESULT4]], %[[RESULT3]] : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
// CHECK: %[[RESULT6:.*]] = tt.load %[[RESULT5]] : tensor<64x1x!tt.ptr<f32>>
tt.func @tensor_without_offset_producer_test(%arg0: !tt.ptr<f16>, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>) -> tensor<64x1xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
  // COM: %0 is a tensor of ptr without offset.
  %1 = tt.addptr %0, %arg2 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
  %2 = tt.bitcast %1 : tensor<64x!tt.ptr<f16>> -> tensor<64x!tt.ptr<f32>>
  %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64x!tt.ptr<f32>> -> tensor<1x64x!tt.ptr<f32>>
  %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<1x64x!tt.ptr<f32>> -> tensor<64x1x!tt.ptr<f32>>
  // COM: End of test.
  %5 = tt.load %4 : tensor<64x1x!tt.ptr<f32>>
  tt.return %5 : tensor<64x1xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_ptr_with_offset_producer_test
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[RESULT1:.*]] = arith.addi %arg5, %c1_i32 : i32
// CHECK: %[[RESULT2:.*]] = arith.addi %arg6, %c1_i32 : i32
// CHECK: %[[RESULT3:.*]] = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%[[RESULT1]], %[[RESULT2]]] {order = array<i32: 0, 1>} : <tensor<32x128xf32>>
// CHECK: %[[RESULT4:.*]] = tt.load %[[RESULT3]] : !tt.ptr<tensor<32x128xf32>>
tt.func @tensor_ptr_with_offset_producer_test(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i32, %arg6: i32) -> tensor<32x128xf32> {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6] {order = array<i32: 0, 1>} : <tensor<32x128xf32>>
  %1 = tt.advance %0, [%c1_i32, %c1_i32] : <tensor<32x128xf32>>
  // COM: %1 is a tensor ptr with offset.
  %2 = tt.advance %1, [%c0_i32, %c0_i32] : <tensor<32x128xf32>>
  // COM: End of test.
  %3 = tt.load %2 : !tt.ptr<tensor<32x128xf32>>
  tt.return %3 : tensor<32x128xf32>
}

// -----
// CHECK-LABEL: tt.func @scalar_offset_type_different_test
// CHECK: %[[RESULT0:.*]] = arith.extsi %arg1 : i32 to i64
// CHECK: %[[RESULT1:.*]] = arith.addi %[[RESULT0]], %arg2 : i64
// CHECK: %[[RESULT2:.*]] = arith.extsi %arg1 : i32 to i64
// CHECK: %[[RESULT3:.*]] = arith.addi %[[RESULT1]], %[[RESULT2]] : i64
// CHECK: %[[RESULT4:.*]] = tt.addptr %arg0, %[[RESULT3]] : !tt.ptr<f32>, i64
tt.func @scalar_offset_type_different_test(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i64) -> f32 {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  // COM: %0 is a scalar ptr with offset.
  %1 = tt.addptr %0, %arg2 : !tt.ptr<f32>, i64
  %2 = tt.addptr %1, %arg1 : !tt.ptr<f32>, i32
  // COM: End of test.
  %3 = tt.load %2 : !tt.ptr<f32>
  tt.return %3 : f32
}

// -----
// CHECK-LABEL: tt.func @tensor_offset_type_different_test
// CHECK: %[[RESULT0:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT1:.*]] = arith.extsi %arg1 : tensor<1x64xi32> to tensor<1x64xi64>
// CHECK: %[[RESULT2:.*]] = arith.addi %[[RESULT1]], %arg2 : tensor<1x64xi64>
// CHECK: %[[RESULT3:.*]] = arith.extsi %arg1 : tensor<1x64xi32> to tensor<1x64xi64>
// CHECK: %[[RESULT4:.*]] = arith.addi %[[RESULT2]], %[[RESULT3]] : tensor<1x64xi64>
// CHECK: %[[RESULT5:.*]] = tt.addptr %[[RESULT0]], %[[RESULT4]] : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi64>
tt.func @tensor_offset_type_different_test(%arg0: !tt.ptr<f32>, %arg1: tensor<1x64xi32>, %arg2: tensor<1x64xi64>) -> tensor<1x64xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  // COM: %1 is a tensor of ptr with offset.
  %2 = tt.addptr %1, %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi64>
  %3 = tt.addptr %2, %arg1 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  // COM: End of test.
  %4 = tt.load %3 : tensor<1x64x!tt.ptr<f32>>
  tt.return %4 : tensor<1x64xf32>
}

// -----
// CHECK-LABEL: tt.func @decomposable_producer_test
// CHECK: %[[RESULT0:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT1:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT2:.*]] = tt.addptr %[[RESULT0]], %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
// CHECK: %[[RESULT3:.*]] = tt.addptr %[[RESULT1]], %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
// CHECK: %[[RESULT4:.*]] = tt.cat %[[RESULT2]], %[[RESULT3]] : tensor<1x64x!tt.ptr<f32>> -> tensor<2x64x!tt.ptr<f32>>
// CHECK: %[[RESULT5:.*]] = tt.addptr %[[RESULT4]], %arg3 : tensor<2x64x!tt.ptr<f32>>, tensor<2x64xi32>
tt.func @decomposable_producer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1x64xi32>, %arg3: tensor<2x64xi32>) -> tensor<2x64xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %2 = tt.addptr %0, %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  %3 = tt.addptr %1, %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  %4 = tt.cat %2, %3 : tensor<1x64x!tt.ptr<f32>> -> tensor<2x64x!tt.ptr<f32>>
  // COM: %4 is a decomposable tensor of ptr with offset.
  %5 = tt.addptr %4, %arg3 : tensor<2x64x!tt.ptr<f32>>, tensor<2x64xi32>
  // COM: End of test.
  %6 = tt.load %5 : tensor<2x64x!tt.ptr<f32>>
  tt.return %6 : tensor<2x64xf32>
}

// -----
// CHECK-LABEL: tt.func @scalar_with_offset_consumer_test
// CHECK: %[[RESULT0:.*]] = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
// CHECK: %[[RESULT1:.*]] = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
// CHECK: %[[RESULT2:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[RESULT0]], %arg2, %arg3 : (!tt.ptr<f32>, f32, i1) -> f32
// CHECK: %[[RESULT3:.*]] = tt.atomic_cas acq_rel, gpu, %[[RESULT0]], %arg1, %c1_i32 : (!tt.ptr<f32>, i32, i32) -> f32
// CHECK: %[[RESULT4:.*]] = arith.addf %[[RESULT2]], %[[RESULT3]] : f32
// CHECK: %[[RESULT5:.*]] = tt.extern_elementwise %[[RESULT0]], %[[RESULT1]] {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK: %[[RESULT6:.*]] = tt.extern_elementwise %[[RESULT0]], %[[RESULT1]] {libname = "libdevice", libpath = "", pure = false, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK: tt.print "ptr: " {hex = false} : %[[RESULT0]] : !tt.ptr<f32>
// CHECK: %[[RESULT7:.*]] = tt.load %[[RESULT0]] : !tt.ptr<f32>
// CHECK: tt.store %[[RESULT5]], %[[RESULT4]] : !tt.ptr<f32>
// CHECK: %[[RESULT8:.*]] = tt.ptr_to_int %[[RESULT0]] : !tt.ptr<f32> -> i64
// CHECK: tt.store %[[RESULT6]], %[[RESULT4]] : !tt.ptr<f32>
// CHECK: %[[RESULT9:.*]] = arith.bitcast %[[RESULT8]] : i64 to f64
// CHECK: %[[RESULT10:.*]] = arith.truncf %[[RESULT9]] : f64 to f32
// CHECK: %[[RESULT11:.*]] = arith.addf %[[RESULT10]], %[[RESULT7]] : f32
tt.func @scalar_with_offset_consumer_test(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: f32, %arg3: i1) -> f32 {
  %c1_i32 = arith.constant 1 : i32
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  %1 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
  // COM: %0, %1 are scalar ptr with offset.
  %2 = "tt.atomic_rmw"(%0, %arg2, %arg3) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, i1) -> f32
  %3 = "tt.atomic_cas"(%0, %arg1, %c1_i32) {sem = 4 : i32, scope = 1 : i32} : (!tt.ptr<f32>, i32, i32) -> f32
  %4 = arith.addf %2, %3 : f32
  %5 = tt.extern_elementwise %0, %1 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
  %6 = tt.extern_elementwise %0, %1 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = false} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
  tt.print "ptr: " {hex = false} : %0 : !tt.ptr<f32>
  %7 = tt.load %0 : !tt.ptr<f32>
  tt.store %5, %4 : !tt.ptr<f32>
  %8 = tt.ptr_to_int %0 : !tt.ptr<f32> -> i64
  // COM: End of test.
  tt.store %6, %4 : !tt.ptr<f32>
  %9 = arith.bitcast %8 : i64 to f64
  %10 = arith.truncf %9 : f64 to f32
  %11 = arith.addf %10, %7 : f32
  tt.return %11 : f32
}

// -----
// CHECK-LABEL: tt.func @scalar_without_offset_consumer_test
// CHECK: %[[RESULT0:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg3 : (!tt.ptr<f32>, f32, i1) -> f32
// CHECK: %[[RESULT1:.*]] = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %[[RESULT0]] : (!tt.ptr<f32>, f32, f32) -> f32
// CHECK: %[[RESULT2:.*]] = tt.extern_elementwise %arg0, %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK: %[[RESULT3:.*]] = tt.extern_elementwise %arg0, %arg0 {libname = "libdevice", libpath = "", pure = false, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK: tt.print "ptr: " {hex = false} : %arg0 : !tt.ptr<f32>
// CHECK: %[[RESULT4:.*]] = tt.load %arg0 : !tt.ptr<f32>
// CHECK: tt.store %[[RESULT2]], %[[RESULT1]] : !tt.ptr<f32>
// CHECK: %[[RESULT5:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
// CHECK: tt.store %[[RESULT3]], %[[RESULT1]] : !tt.ptr<f32>
// CHECK: %[[RESULT6:.*]] = arith.bitcast %[[RESULT5]] : i64 to f64
tt.func @scalar_without_offset_consumer_test(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32, %arg3: i1) -> f32 {
  // COM: %arg0 is a scalar ptr without offset.
  %0 = "tt.atomic_rmw"(%arg0, %arg2, %arg3) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, i1) -> f32
  %1 = "tt.atomic_cas"(%arg0, %arg1, %0) {sem = 4 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, f32) -> f32
  %2 = tt.extern_elementwise %arg0, %arg0 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
  %3 = tt.extern_elementwise %arg0, %arg0 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = false} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
  tt.print "ptr: " {hex = false} : %arg0 : !tt.ptr<f32>
  %4 = tt.load %arg0 : !tt.ptr<f32>
  tt.store %2, %1 : !tt.ptr<f32>
  %5 = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
  // COM: End of test.
  tt.store %3, %1 : !tt.ptr<f32>
  %6 = arith.bitcast %5 : i64 to f64
  %7 = arith.truncf %6 : f64 to f32
  %8 = arith.addf %7, %4 : f32
  tt.return %8 : f32
}

// -----
// CHECK-LABEL: tt.func @tensor_with_offset_consumer_test
// CHECK: %[[RESULT0:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT1:.*]] = tt.addptr %[[RESULT0]], %arg1 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
// CHECK: %[[RESULT2:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[RESULT1]], %arg2, %arg3 : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xi1>) -> tensor<1x64xf32>
// CHECK: %[[RESULT3:.*]] = tt.atomic_cas acq_rel, gpu, %[[RESULT1]], %arg2, %[[RESULT2]] : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[RESULT4:.*]] = tt.extern_elementwise %[[RESULT1]], %[[RESULT1]] {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT5:.*]] = tt.extern_elementwise %[[RESULT1]], %[[RESULT1]] {libname = "libdevice", libpath = "", pure = false, symbol = "__cn_vector_min"} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT6:.*]] = tt.cat %[[RESULT1]], %[[RESULT1]] : tensor<1x64x!tt.ptr<f32>> -> tensor<2x!tt.ptr<f32>>
// CHECK: %[[RESULT7:.*]] = tt.load %[[RESULT1]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT8:.*]] = "tt.reduce"(%[[RESULT1]]) <{axis = 1 : i32}> ({
// CHECK: ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
// CHECK:   %[[RESULT13:.*]] = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK:   tt.reduce.return %[[RESULT13]] : !tt.ptr<f32>
// CHECK: }) : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[RESULT9:.*]] = "tt.scan"(%[[RESULT1]]) <{axis = 1 : i32, reverse = false}> ({
// CHECK: ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
// CHECK:   %[[RESULT13:.*]] = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK:   tt.scan.return %[[RESULT13]] : !tt.ptr<f32>
// CHECK: }) : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT10:.*]] = tt.reshape %arg1 {allow_reorder = false} : tensor<1x64xi32> -> tensor<4x16xi32>
// CHECK: %[[RESULT11:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
// CHECK: %[[RESULT12:.*]] = tt.addptr %[[RESULT11]], %[[RESULT10]] : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
// CHECK: tt.store %[[RESULT1]], %[[RESULT3]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: tt.print "ptr: " {hex = false} : %[[RESULT1]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: tt.assert %[[RESULT1]], "message", "file", "line", 87 : tensor<1x64x!tt.ptr<f32>>
tt.func @tensor_with_offset_consumer_test(%arg0: !tt.ptr<f32>, %arg1: tensor<1x64xi32>, %arg2: tensor<1x64xf32>, %arg3: tensor<1x64xi1>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  // COM: %1 is a tensor of ptr with offset.
  %2 = "tt.atomic_rmw"(%1, %arg2, %arg3) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xi1>) -> tensor<1x64xf32>
  %3 = "tt.atomic_cas"(%1, %arg2, %2) {sem = 4 : i32, scope = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
  %4 = tt.extern_elementwise %1, %1 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %5 = tt.extern_elementwise %1, %1 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = false} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %6 = tt.cat %1, %1 : tensor<1x64x!tt.ptr<f32>> -> tensor<2x!tt.ptr<f32>>
  %7 = tt.load %1 : tensor<1x64x!tt.ptr<f32>>
  %8 = "tt.reduce"(%1) ({
  ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
    %11 = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
    tt.reduce.return %11 : !tt.ptr<f32>
  }) {axis = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x!tt.ptr<f32>>
  %9 = "tt.scan"(%1) ({
  ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
    %11 = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
    tt.scan.return %11 : !tt.ptr<f32>
  }) {axis = 1 : i32, reverse = false} : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %10 = tt.reshape %1 {allow_reorder = false} : tensor<1x64x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
  tt.store %1, %3 : tensor<1x64x!tt.ptr<f32>>
  tt.print "ptr: " {hex = false} : %1 : tensor<1x64x!tt.ptr<f32>>
  tt.assert %1, "message", "file", "line", 87 : tensor<1x64x!tt.ptr<f32>>
  // COM: End of test.
  tt.print "%2" {hex = false} : %2 : tensor<1x64xf32>
  tt.print "%3" {hex = false} : %3 : tensor<1x64xf32>
  tt.print "%4" {hex = false} : %4 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%5" {hex = false} : %5 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%6" {hex = false} : %6 : tensor<2x!tt.ptr<f32>>
  tt.print "%7" {hex = false} : %7 : tensor<1x64xf32>
  tt.print "%8" {hex = false} : %8 : tensor<1x!tt.ptr<f32>>
  tt.print "%9" {hex = false} : %9 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%10" {hex = false} : %10 : tensor<4x16x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: tt.func @tensor_without_offset_consumer_test
// CHECK: %[[RESULT0:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT1:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[RESULT0]], %arg2, %arg3 : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xi1>) -> tensor<1x64xf32>
// CHECK: %[[RESULT2:.*]] = tt.atomic_cas acq_rel, gpu, %[[RESULT0]], %arg2, %[[RESULT1]] : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
// CHECK: %[[RESULT3:.*]] = tt.extern_elementwise %[[RESULT0]], %[[RESULT0]] {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT4:.*]] = tt.extern_elementwise %[[RESULT0]], %[[RESULT0]] {libname = "libdevice", libpath = "", pure = false, symbol = "__cn_vector_min"} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT5:.*]] = tt.cat %[[RESULT0]], %[[RESULT0]] : tensor<1x64x!tt.ptr<f32>> -> tensor<2x!tt.ptr<f32>>
// CHECK: %[[RESULT6:.*]] = tt.load %[[RESULT0]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT7:.*]] = "tt.reduce"(%[[RESULT0]]) <{axis = 1 : i32}> ({
// CHECK: ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
// CHECK:   %[[RESULT10:.*]] = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK:   tt.reduce.return %[[RESULT10]] : !tt.ptr<f32>
// CHECK: }) : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[RESULT8:.*]] = "tt.scan"(%[[RESULT0]]) <{axis = 1 : i32, reverse = false}> ({
// CHECK: ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
// CHECK:   %[[RESULT10:.*]] = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", pure = true, symbol = "__cn_vector_min"} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
// CHECK:   tt.scan.return %[[RESULT10]] : !tt.ptr<f32>
// CHECK: }) : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[RESULT9:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
// CHECK: tt.store %[[RESULT0]], %[[RESULT2]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: tt.print "ptr: " {hex = false} : %[[RESULT0]] : tensor<1x64x!tt.ptr<f32>>
// CHECK: tt.assert %[[RESULT0]], "message", "file", "line", 87 : tensor<1x64x!tt.ptr<f32>>
tt.func @tensor_without_offset_consumer_test(%arg0: !tt.ptr<f32>, %arg1: tensor<1x64xi32>, %arg2: tensor<1x64xf32>, %arg3: tensor<1x64xi1>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  // COM: %0 is a tensor of ptr without offset.
  %1 = "tt.atomic_rmw"(%0, %arg2, %arg3) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xi1>) -> tensor<1x64xf32>
  %2 = "tt.atomic_cas"(%0, %arg2, %1) {sem = 4 : i32, scope = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
  %3 = tt.extern_elementwise %0, %0 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %4 = tt.extern_elementwise %0, %0 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = false} : (tensor<1x64x!tt.ptr<f32>>, tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %5 = tt.cat %0, %0 : tensor<1x64x!tt.ptr<f32>> -> tensor<2x!tt.ptr<f32>>
  %6 = tt.load %0 : tensor<1x64x!tt.ptr<f32>>
  %7 = "tt.reduce"(%0) ({
  ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
    %10 = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
    tt.reduce.return %10 : !tt.ptr<f32>
  }) {axis = 1 : i32} : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x!tt.ptr<f32>>
  %8 = "tt.scan"(%0) ({
  ^bb0(%arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>):
    %10 = tt.extern_elementwise %arg4, %arg5 {libname = "libdevice", libpath = "", symbol = "__cn_vector_min", pure = true} : (!tt.ptr<f32>, !tt.ptr<f32>) -> !tt.ptr<f32>
    tt.scan.return %10 : !tt.ptr<f32>
  }) {axis = 1 : i32, reverse =false} : (tensor<1x64x!tt.ptr<f32>>) -> tensor<1x64x!tt.ptr<f32>>
  %9 = tt.reshape %0 {allow_reorder = false} : tensor<1x64x!tt.ptr<f32>> -> tensor<4x16x!tt.ptr<f32>>
  tt.store %0, %2 : tensor<1x64x!tt.ptr<f32>>
  tt.print "ptr: " {hex = false} : %0 : tensor<1x64x!tt.ptr<f32>>
  tt.assert %0, "message", "file", "line", 87 : tensor<1x64x!tt.ptr<f32>>
  // COM: End of test.
  tt.print "%2" {hex = false} : %1 : tensor<1x64xf32>
  tt.print "%3" {hex = false} : %2 : tensor<1x64xf32>
  tt.print "%4" {hex = false} : %3 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%5" {hex = false} : %4 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%6" {hex = false} : %5 : tensor<2x!tt.ptr<f32>>
  tt.print "%7" {hex = false} : %6 : tensor<1x64xf32>
  tt.print "%8" {hex = false} : %7 : tensor<1x!tt.ptr<f32>>
  tt.print "%9" {hex = false} : %8 : tensor<1x64x!tt.ptr<f32>>
  tt.print "%10" {hex = false} : %9 : tensor<4x16x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: tt.func @tensor_ptr_with_offset_consumer_test
// CHECK:      %[[ARG2:.*]] = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%0, %1] {order = array<i32: 0, 1>} : <tensor<32x128xf16>>
// CHECK-NEXT: tt.print "%1" {hex = false} : %[[ARG2]] : !tt.ptr<tensor<32x128xf16>>
// CHECK-NEXT: %[[ARG3:.*]] = tt.load %[[ARG2]] : !tt.ptr<tensor<32x128xf16>>
// CHECK-NEXT: tt.store %[[ARG2]], %[[ARG3]] : !tt.ptr<tensor<32x128xf16>>
tt.func @tensor_ptr_with_offset_consumer_test(%arg0: !tt.ptr<f16>, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i32, %arg6: i32) {
  %c1_i32 = arith.constant 1 : i32
  %0 = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6] {order = array<i32: 0, 1>} : <tensor<32x128xf16>>
  %1 = tt.advance %0, [%c1_i32, %c1_i32] : <tensor<32x128xf16>>
  // COM: %1 is a tensor ptr with offset.
  tt.print "%1" {hex = false} : %1 : !tt.ptr<tensor<32x128xf16>>
  %2 = tt.load %1 : !tt.ptr<tensor<32x128xf16>>
  tt.store %1, %2 : !tt.ptr<tensor<32x128xf16>>
  // COM: End of test.
  tt.return
}

// -----
// CHECK-LABEL: tt.func @tensor_ptr_without_offset_consumer_test
// CHECK:      %[[ARG0:.*]] = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6] {order = array<i32: 0, 1>} : <tensor<32x128xf16>>
// CHECK-NEXT: tt.print "%0" {hex = false} : %[[ARG0]] : !tt.ptr<tensor<32x128xf16>>
// CHECK-NEXT: %[[ARG1:.*]] = tt.load %[[ARG0]] : !tt.ptr<tensor<32x128xf16>>
// CHECK-NEXT: tt.store %[[ARG0]], %[[ARG1]] : !tt.ptr<tensor<32x128xf16>>
tt.func @tensor_ptr_without_offset_consumer_test(%arg0: !tt.ptr<f16>, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i32, %arg6: i32) {
  %0 = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %arg4], [%arg5, %arg6] {order = array<i32: 0, 1>} : <tensor<32x128xf16>>
  // COM: %0 is a tensor ptr without offset.
  tt.print "%0" {hex = false} : %0 : !tt.ptr<tensor<32x128xf16>>
  %1 = tt.load %0 : !tt.ptr<tensor<32x128xf16>>
  tt.store %0, %1 : !tt.ptr<tensor<32x128xf16>>
  // COM: End of test.
  tt.return
}

// -----
// CHECK-LABEL: tt.func @tensor_base_ptr_type_different
// CHECK:      %[[ARG1:.*]] = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK-NEXT: %[[ARG2:.*]] = tt.bitcast %[[ARG1]] : !tt.ptr<i8> -> !tt.ptr<f32>
// CHECK-NEXT: %[[ARG3:.*]] = tt.splat %[[ARG2]] : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT: %[[ARG4:.*]] = tt.addptr %[[ARG3]], %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
tt.func @tensor_base_ptr_type_different(%arg0: !tt.ptr<i1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
  %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
  %4 = tt.bitcast %3 : tensor<128x!tt.ptr<i8>> -> tensor<128x!tt.ptr<f32>>
  %5 = tt.load %4 : tensor<128x!tt.ptr<f32>>
  tt.return %5 : tensor<128xf32>
}

// -----
// CHECK-LABEL: tt.func @decomposable_consumer_test
// CHECK:      %[[ARG0:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK-NEXT: %[[ARG1:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
// CHECK-NEXT: %[[ARG2:.*]] = tt.addptr %[[ARG0]], %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
// CHECK-NEXT: %[[ARG3:.*]] = tt.addptr %[[ARG1]], %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
// CHECK-NEXT: %[[ARG4:.*]] = tt.cat %[[ARG2]], %[[ARG3]] : tensor<1x64x!tt.ptr<f32>> -> tensor<2x64x!tt.ptr<f32>>
tt.func @decomposable_consumer_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1x64xi32>) -> tensor<2x64xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %2 = tt.addptr %0, %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  %3 = tt.addptr %1, %arg2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  %4 = tt.cat %2, %3 : tensor<1x64x!tt.ptr<f32>> -> tensor<2x64x!tt.ptr<f32>>
  // COM: %4 is a decomposable tensor of ptr with offset.
  %5 = tt.load %4 : tensor<2x64x!tt.ptr<f32>>
  // COM: End of test.
  tt.return %5 : tensor<2x64xf32>
}

// -----
// CHECK-LABEL: tt.func @for_iter_arg_no_use_test
// CHECK:      %[[ARG0:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT: %[[ARG1:.*]]:2 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %[[ARG0]], %arg3 = %[[ARG0]]) -> (tensor<32xi32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[INDEX_CAST:.*]] = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:   %[[RANGE:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[SPLAT:.*]] = tt.splat %[[INDEX_CAST]] : i32 -> tensor<32xi32>
// CHECK-NEXT:   %[[ADDI:.*]] = arith.addi %[[RANGE]], %[[SPLAT]] : tensor<32xi32>
// CHECK-NEXT:   scf.yield %[[ADDI]], %arg3 : tensor<32xi32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[ARG2:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[ARG3:.*]] = tt.addptr %[[ARG2]], %[[ARG1]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @for_iter_arg_no_use_test(%arg0: !tt.ptr<f32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1:2 = scf.for %arg1 = %c0 to %c128 step %c32 iter_args(%arg2 = %0, %arg3 = %0) -> (tensor<32xi32>, tensor<32xi32>) {
    %5 = arith.index_cast %arg1 : index to i32
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %7 = tt.splat %5 : i32 -> tensor<32xi32>
    %8 = arith.addi %6, %7 : tensor<32xi32>
    scf.yield %8, %arg3 : tensor<32xi32>, tensor<32xi32>
  }
  %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %3 = tt.addptr %2, %1#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %4 = tt.load %3 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_if_two_region_base_ptr_different_test
// CHECK: %[[RESULT1:.*]]{{:.*}} = scf.if %arg1 -> (!tt.ptr<f32>, tensor<32xi32>) {
// CHECK: %[[RESULT5:.*]] = arith.addi %0, %cst : tensor<32xi32>
// CHECK:     scf.yield %arg0, %[[RESULT5]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK:   } else {
// CHECK:     scf.yield %arg2, %cst_0 : !tt.ptr<f32>, tensor<32xi32>
// CHECK:   }
// CHECK: %[[RESULT6:.*]] = tt.splat %[[RESULT1]]#0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK: %[[RESULT7:.*]] = tt.addptr %[[RESULT6]], %[[RESULT1]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @tensor_if_two_region_base_ptr_different_test(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<42> : tensor<32xi32>
  %cst_0 = arith.constant dense<2> : tensor<32xi32>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %4 = scf.if %arg1 -> (tensor<32x!tt.ptr<f32>>) {
    %6 = tt.addptr %2, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %6 : tensor<32x!tt.ptr<f32>>
  } else {
    %6 = tt.addptr %3, %cst_0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %6 : tensor<32x!tt.ptr<f32>>
  }
  %5 = tt.load %4 : tensor<32x!tt.ptr<f32>>
  tt.return %5 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_ptr_if_two_region_base_ptr_different_test
// CHECK:      %[[ARG1:.*]]:8 = scf.if %arg1 -> (tensor<32x32xf32>, !tt.ptr<f32>, i64, i64, i64, i64, i32, i32) {
// CHECK-NEXT:   %[[ARG2:.*]] = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%arg2, %arg3] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
// CHECK-NEXT:   %[[ARG3:.*]] = tt.load %[[ARG2]] : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:   scf.yield %[[ARG3]], %arg0, %c128_i64, %c64_i64, %c1_i64, %c1_i64, %arg2, %arg3 : tensor<32x32xf32>, !tt.ptr<f32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[ARG4:.*]] = tt.make_tensor_ptr %arg4, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%arg2, %arg3] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
// CHECK-NEXT:   %[[ARG5:.*]] = tt.load %[[ARG4]] : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:   scf.yield %[[ARG5]], %arg4, %c128_i64, %c64_i64, %c1_i64, %c1_i64, %arg2, %arg3 : tensor<32x32xf32>, !tt.ptr<f32>, i64, i64, i64, i64, i32, i32
// CHECK-NEXT: }
// CHECK-NEXT: %1 = tt.make_tensor_ptr %[[ARG1]]#1, [%[[ARG1]]#2, %[[ARG1]]#3], [%[[ARG1]]#4, %[[ARG1]]#5], [%[[ARG1]]#6, %[[ARG1]]#7] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
tt.func @tensor_ptr_if_two_region_base_ptr_different_test(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: i32, %arg3: i32, %arg4: !tt.ptr<f32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %1 = tt.make_tensor_ptr %arg4, [%c128_i64, %c64_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x32xf32>>
  %2:2 = scf.if %arg1 -> (!tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>) {
    %4 = tt.advance %0, [%arg2, %arg3] : <tensor<32x32xf32>>
    %5 = tt.load %4 : !tt.ptr<tensor<32x32xf32>>
    scf.yield %4, %5 : !tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>
  } else {
    %4 = tt.advance %1, [%arg2, %arg3] : <tensor<32x32xf32>>
    %5 = tt.load %4 : !tt.ptr<tensor<32x32xf32>>
    scf.yield %4, %5 : !tt.ptr<tensor<32x32xf32>>, tensor<32x32xf32>
  }
  %3 = tt.load %2#0 : !tt.ptr<tensor<32x32xf32>>
  tt.return %3, %2#1 : tensor<32x32xf32>, tensor<32x32xf32>
}

// -----
// CHECK-LABEL: tt.func @for_nested_while
// CHECK: %[[TEMP7:.*]]:3 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %0, %arg4 = %arg0, %arg5 = %0) -> (tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>) {
// CHECK:      %[[TEMP0:.*]] = tt.splat %5 : i32 -> tensor<32xi32>
// CHECK-NEXT: %[[TEMP1:.*]] = arith.addi %6, %[[TEMP0]] : tensor<32xi32>
// CHECK-NEXT: %[[TEMP2:.*]] = arith.addi %[[TEMP1]], %arg5 : tensor<32xi32>
// CHECK-NEXT: %[[TEMP3:.*]]:2 = scf.while (%arg6 = %arg4, %arg7 = %[[TEMP2]]) : (!tt.ptr<f32>, tensor<32xi32>) -> (!tt.ptr<f32>, tensor<32xi32>) {
// CHECK-NEXT:   %[[TEMP4:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:   %[[TEMP5:.*]] = arith.addi %[[TEMP4]], %arg7 : tensor<32xi32>
// CHECK-NEXT:   scf.condition(%arg1) %arg6, %[[TEMP5]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%arg6: !tt.ptr<f32>, %arg7: tensor<32xi32>):
// CHECK-NEXT:   %[[TEMP6:.*]] = arith.addi %arg7, %cst : tensor<32xi32>
// CHECK-NEXT:   scf.yield %arg6, %[[TEMP6]] : !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[TEMP1]], %[[TEMP3]]#0, %[[TEMP3]]#1 : tensor<32xi32>, !tt.ptr<f32>, tensor<32xi32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[TEMP8:.*]] = tt.splat %[[TEMP7]]#1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[TEMP9:.*]] = tt.addptr %[[TEMP8]], %[[TEMP7]]#2 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
tt.func @for_nested_while(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant dense<2> : tensor<32xi32>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3:2 = scf.for %arg2 = %c0 to %c128 step %c32 iter_args(%arg3 = %0, %arg4 = %2) -> (tensor<32xi32>, tensor<32x!tt.ptr<f32>>) {
    %5 = arith.index_cast %arg2 : index to i32
    %6 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %7 = tt.splat %5 : i32 -> tensor<32xi32>
    %8 = arith.addi %6, %7 : tensor<32xi32>
    %9 = tt.addptr %arg4, %8 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %10 = scf.while (%arg5 = %9) : (tensor<32x!tt.ptr<f32>>) -> tensor<32x!tt.ptr<f32>> {
      %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %12 = tt.addptr %arg5, %11 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      scf.condition(%arg1) %12 : tensor<32x!tt.ptr<f32>>
    } do {
    ^bb0(%arg5: tensor<32x!tt.ptr<f32>>):
      %11 = tt.addptr %arg5, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      scf.yield %11 : tensor<32x!tt.ptr<f32>>
    }
    scf.yield %8, %10 : tensor<32xi32>, tensor<32x!tt.ptr<f32>>
  }
  %4 = tt.load %3#1 : tensor<32x!tt.ptr<f32>>
  tt.return %4 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @tensor_if_two_region_offset_type_different_test
// CHECK:      %[[ARG1:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT: %[[ARG2:.*]]:2 = scf.if %arg1 -> (!tt.ptr<f32>, tensor<32xi64>) {
// CHECK-NEXT:   %[[ARG3:.*]] = arith.addi %[[ARG1]], %cst : tensor<32xi32>
// CHECK-NEXT:   %[[ARG4:.*]] = arith.extsi %[[ARG3]] : tensor<32xi32> to tensor<32xi64>
// CHECK-NEXT:   scf.yield %arg0, %[[ARG4]] : !tt.ptr<f32>, tensor<32xi64>
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %arg2, %cst_0 : !tt.ptr<f32>, tensor<32xi64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[ARG5:.*]] = tt.splat %[[ARG2]]#0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT: %[[ARG6:.*]] = tt.addptr %[[ARG5]], %[[ARG2]]#1 : tensor<32x!tt.ptr<f32>>, tensor<32xi64>
tt.func @tensor_if_two_region_offset_type_different_test(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: !tt.ptr<f32>) -> tensor<32xf32> {
  %cst = arith.constant dense<42> : tensor<32xi32>
  %cst_0 = arith.constant dense<2> : tensor<32xi64>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %3 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %4 = scf.if %arg1 -> (tensor<32x!tt.ptr<f32>>) {
    %6 = tt.addptr %2, %cst : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    scf.yield %6 : tensor<32x!tt.ptr<f32>>
  } else {
    %6 = tt.addptr %3, %cst_0 : tensor<32x!tt.ptr<f32>>, tensor<32xi64>
    scf.yield %6 : tensor<32x!tt.ptr<f32>>
  }
  %5 = tt.load %4 : tensor<32x!tt.ptr<f32>>
  tt.return %5 : tensor<32xf32>
}

// -----
// CHECK-LABEL: tt.func @control_flow_cond_br_with_three_blocks_test
// CHECK:      cf.cond_br %arg0, ^bb1(%arg1, %cst : !tt.ptr<f32>, tensor<128xi32>), ^bb2(%arg2, %cst_0 : !tt.ptr<f32>, tensor<128xi64>)
// CHECK:    ^bb1(%0: !tt.ptr<f32>, %1: tensor<128xi32>):  // pred: ^bb0
// CHECK-NEXT:   %[[ARG1:.*]] = arith.addi %1, %cst : tensor<128xi32>
// CHECK-NEXT:   %[[ARG2:.*]] = arith.extsi %[[ARG1]] : tensor<128xi32> to tensor<128xi64>
// CHECK-NEXT:   cf.br ^bb3(%0, %[[ARG2]] : !tt.ptr<f32>, tensor<128xi64>)
// CHECK:    ^bb2(%4: !tt.ptr<f32>, %5: tensor<128xi64>):  // pred: ^bb0
// CHECK-NEXT:   %[[ARG3:.*]] = arith.addi %5, %cst_0 : tensor<128xi64>
// CHECK-NEXT:   cf.br ^bb3(%4, %[[ARG3]] : !tt.ptr<f32>, tensor<128xi64>)
// CHECK:    ^bb3(%7: !tt.ptr<f32>, %8: tensor<128xi64>):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:   %[[ARG4:.*]] = tt.splat %7 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:   %[[ARG5:.*]] = tt.addptr %[[ARG4]], %8 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
tt.func @control_flow_cond_br_with_three_blocks_test(%arg0: i1, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) -> tensor<128xf32> {
  %cst = arith.constant dense<86> : tensor<128xi32>
  %cst_0 = arith.constant dense<2> : tensor<128xi64>
  %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %2 = tt.addptr %0, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %3 = tt.addptr %1, %cst_0 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
  cf.cond_br %arg0, ^bb1(%2 : tensor<128x!tt.ptr<f32>>), ^bb2(%3 : tensor<128x!tt.ptr<f32>>)
^bb1(%4: tensor<128x!tt.ptr<f32>>):  // pred: ^bb0
  %5 = tt.addptr %4, %cst : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  cf.br ^bb3(%5 : tensor<128x!tt.ptr<f32>>)
^bb2(%6: tensor<128x!tt.ptr<f32>>):  // pred: ^bb0
  %7 = tt.addptr %6, %cst_0 : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
  cf.br ^bb3(%7 : tensor<128x!tt.ptr<f32>>)
^bb3(%8: tensor<128x!tt.ptr<f32>>):  // 2 preds: ^bb1, ^bb2
  %9 = tt.load %8 : tensor<128x!tt.ptr<f32>>
  tt.return %9 : tensor<128xf32>
}

// -----
// CHECK-LABEL: tt.func @for_addptr_use_splat_test
// CHECK-SAME: %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32)
// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1x1xi32>
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
// CHECK: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: %[[ARG4:.*]] = arith.addi %[[ARG3]], %[[ARG2]] : i32
// CHECK: %[[ARG5:.*]] = tt.addptr %[[ARG1]], %[[ARG4]] : !tt.ptr<f32>, i32
// CHECK: %[[ARG6:.*]] = arith.muli %[[ARG2]], %[[ARG3]] : i32
// CHECK: %[[ARG7:.*]] = tt.splat %[[ARG6]] : i32 -> tensor<1x1xi32>
// CHECK: %[[ARG8:.*]]:2 = scf.for %[[ARG9:.*]] = %[[C0_I32]] to %[[C2_I32]] step %[[C1_I32]] iter_args(%[[ITER_PTR:.*]] = %[[ARG5]], %[[ARG10:.*]] = %[[CST]]) -> (!tt.ptr<f32>, tensor<1x1xi32>)  : i32 {
// CHECK:   %[[ARG11:.*]] = tt.splat %[[ITER_PTR]] : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK:   %[[ARG12:.*]] = tt.addptr %[[ARG11]], %[[ARG10]] : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
// CHECK:   tt.store %[[ARG12]], %[[CST_0]] : tensor<1x1x!tt.ptr<f32>>
// CHECK:   %[[ARG13:.*]] = arith.addi %[[ARG7]], %[[ARG10]] : tensor<1x1xi32>
// CHECK:   scf.yield %[[ITER_PTR]], %[[ARG13]] : !tt.ptr<f32>, tensor<1x1xi32>
// CHECK: }
// CHECK: %[[ARG14:.*]] = tt.splat %[[ARG8]]#0 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK: %[[ARG15:.*]] = tt.addptr %[[ARG14]], %[[ARG8]]#1 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
// CHECK: tt.return %[[ARG15]] : tensor<1x1x!tt.ptr<f32>
tt.func @for_addptr_use_splat_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) -> tensor<1x1x!tt.ptr<f32>> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1xf32>

  %1 = tt.addptr %arg1, %arg2 : !tt.ptr<f32>, i32
  %2 = tt.addptr %1, %arg3 : !tt.ptr<f32>, i32
  %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>

  %4 = arith.muli %arg2, %arg3 : i32
  %5 = tt.splat %4 : i32 -> tensor<1x1xi32>
  %6 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %3) -> (tensor<1x1x!tt.ptr<f32>>)  : i32 {
    tt.store %arg5, %cst : tensor<1x1x!tt.ptr<f32>>
    %7 = tt.addptr %arg5, %5 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
    scf.yield %7 : tensor<1x1x!tt.ptr<f32>>
  }
  tt.return %6 : tensor<1x1x!tt.ptr<f32>>
}

// -----
// CHECK-LABEL: tt.func @if_addptr_use_splat_test
// CHECK-SAME: %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i1)
// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1x1xi32>
// CHECK: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: %[[ARG5:.*]] = arith.addi %[[ARG3]], %[[ARG2]] : i32
// CHECK: %[[ARG6:.*]] = tt.addptr %[[ARG1]], %[[ARG5]] : !tt.ptr<f32>, i32
// CHECK: %[[ARG7:.*]] = tt.splat %[[ARG6]] : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK: %[[ARG8:.*]] = arith.muli %[[ARG2]], %[[ARG3]] : i32
// CHECK: %[[ARG9:.*]] = tt.splat %[[ARG8]] : i32 -> tensor<1x1xi32>
// CHECK: %[[ARG10:.*]] = scf.if %[[ARG4]] -> (tensor<1x1xi32>) {
// CHECK:   tt.store %[[ARG7]], %[[CST_0]] : tensor<1x1x!tt.ptr<f32>>
// CHECK:   scf.yield %[[ARG9]] : tensor<1x1xi32>
// CHECK: } else {
// CHECK:   tt.store %[[ARG7]], %[[CST_0]] : tensor<1x1x!tt.ptr<f32>>
// CHECK:   scf.yield %[[CST]] : tensor<1x1xi32>
// CHECK: }
// CHECK: %[[ARG11:.*]] = tt.splat %[[ARG6]] : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK: %[[ARG12:.*]] = tt.addptr %[[ARG11]], %[[ARG10]] : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
// CHECK: tt.return %[[ARG12]] : tensor<1x1x!tt.ptr<f32>>
tt.func @if_addptr_use_splat_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i1) -> tensor<1x1x!tt.ptr<f32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1xf32>

  %1 = tt.addptr %arg1, %arg2 : !tt.ptr<f32>, i32
  %2 = tt.addptr %1, %arg3 : !tt.ptr<f32>, i32
  %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>

  %4 = arith.muli %arg2, %arg3 : i32
  %5 = tt.splat %4 : i32 -> tensor<1x1xi32>

  %6 = scf.if %arg4 -> (tensor<1x1x!tt.ptr<f32>>) {
    tt.store %3, %cst : tensor<1x1x!tt.ptr<f32>>
    %7 = tt.addptr %3, %5 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
    scf.yield %7 : tensor<1x1x!tt.ptr<f32>>
  } else {
    tt.store %3, %cst : tensor<1x1x!tt.ptr<f32>>
    scf.yield %3 : tensor<1x1x!tt.ptr<f32>>
  }
  tt.return %6 : tensor<1x1x!tt.ptr<f32>>
}

// -----
// CHECK-LABEL: tt.func @while_addptr_use_splat_test
// CHECK-SAME: %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32)
// CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<1x1xi32>
// CHECK: %[[CST_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[CST1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[CST2_I32:.*]] = arith.constant 2 : i32
// CHECK: %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
// CHECK: %[[ARG4:.*]] = arith.addi %[[ARG3]], %[[ARG2]] : i32
// CHECK: %[[ARG5:.*]] = tt.addptr %[[ARG1]], %[[ARG4]] : !tt.ptr<f32>, i32
// CHECK: %[[ARG6:.*]] = arith.muli %[[ARG2]], %[[ARG3]] : i32
// CHECK: %[[ARG7:.*]] = tt.splat %[[ARG6]] : i32 -> tensor<1x1xi32>
// CHECK: %[[ARG8:.*]]:3 = scf.while (%arg4 = %c0_i32, %arg5 = %1, %arg6 = %cst) : (i32, !tt.ptr<f32>, tensor<1x1xi32>) -> (i32, !tt.ptr<f32>, tensor<1x1xi32>) {
// CHECK:   %[[ARG11:.*]] = arith.cmpi slt, %arg4, %c2_i32 : i32
// CHECK:   scf.condition(%[[ARG11]]) %arg4, %arg5, %arg6 : i32, !tt.ptr<f32>, tensor<1x1xi32>
// CHECK: } do {
// CHECK: ^bb0(%arg4: i32, %arg5: !tt.ptr<f32>, %arg6: tensor<1x1xi32>):
// CHECK:   %7 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK:   %8 = tt.addptr %7, %arg6 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
// CHECK:   tt.store %8, %cst_0 : tensor<1x1x!tt.ptr<f32>>
// CHECK:   %9 = arith.addi %3, %arg6 : tensor<1x1xi32>
// CHECK:   %10 = arith.addi %arg4, %c1_i32 : i32
// CHECK:   scf.yield %10, %arg5, %9 : i32, !tt.ptr<f32>, tensor<1x1xi32>
// CHECK: }
// CHECK:  %5 = tt.splat %4#1 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
// CHECK:  %6 = tt.addptr %5, %4#2 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
// CHECK:  tt.return %6 : tensor<1x1x!tt.ptr<f32>>
tt.func @while_addptr_use_splat_test(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) -> tensor<1x1x!tt.ptr<f32>> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1xf32>

  %1 = tt.addptr %arg1, %arg2 : !tt.ptr<f32>, i32
  %2 = tt.addptr %1, %arg3 : !tt.ptr<f32>, i32
  %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>

  %4 = arith.muli %arg2, %arg3 : i32
  %5 = tt.splat %4 : i32 -> tensor<1x1xi32>

  %6, %7 = scf.while (%arg4 = %c0, %arg5 = %3) : (i32, tensor<1x1x!tt.ptr<f32>>) -> (i32, tensor<1x1x!tt.ptr<f32>>) {
    %cond = arith.cmpi slt, %arg4, %c2 : i32
    scf.condition(%cond) %arg4, %arg5 : i32, tensor<1x1x!tt.ptr<f32>>
  } do {
  ^bb0(%arg4: i32, %arg5: tensor<1x1x!tt.ptr<f32>>):
    tt.store %arg5, %cst : tensor<1x1x!tt.ptr<f32>>
    %8 = tt.addptr %arg5, %5 : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
    %arg4_next = arith.addi %arg4, %c1 : i32
    scf.yield %arg4_next, %8 : i32, tensor<1x1x!tt.ptr<f32>>
  }
  tt.return %7 : tensor<1x1x!tt.ptr<f32>>
}
