// RUN: triton-linalg-opt --convert-triton-to-linalg %s -split-input-file | FileCheck %s

// CHECK-LABEL: @func(%arg0: i64)
tt.func @func(%arg0: !tt.ptr<f32>) {
  tt.return
}

// -----
// CHECK-LABEL: @func_ptr_of_ptr(%arg0: i64)
tt.func @func_ptr_of_ptr(%arg0: !tt.ptr<!tt.ptr<f32>>) {
  tt.return
}

// -----
// CHECK-LABEL: @func_tensor_of_ptr(%arg0: tensor<256xi64>)
tt.func @func_tensor_of_ptr(%arg0: tensor<256x!tt.ptr<f32>>) {
  tt.return
}

// -----
// CHECK-LABEL: @broadcast_tensor_int
tt.func @broadcast_tensor_int(%arg0: tensor<256x1xi32>) {
  // CHECK: tensor.collapse_shape  %arg0 {{\[\[0, 1]]}} : tensor<256x1xi32> into tensor<256xi32>
  // CHECK: tensor.empty
  // CHECK: linalg.broadcast
  %0 = tt.broadcast %arg0 : tensor<256x1xi32> -> tensor<256x64xi32>
  tt.return
}

// -----
// CHECK-LABEL: @splat_int
tt.func @splat_int(%arg0: i32) {
  // CHECK: tensor.empty
  // CHECK: linalg.fill
  %0 = tt.splat %arg0 : i32 -> tensor<256xi32>
  tt.return
}

// -----
// CHECK-LABEL: @splat_ptr
// CHECK-SAME:  %[[ARG:.*]]: i64
tt.func @splat_ptr(%arg0: !tt.ptr<f32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: linalg.fill ins(%[[ARG]] : i64) outs(%[[INIT]] : tensor<256xi64>)
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @expand_dims_int
tt.func @expand_dims_int(%arg0: tensor<8xi32>) {
  // CHECK: tensor.expand_shape %arg0 {{\[\[0, 1]]}}
  %0 = tt.expand_dims %arg0 {axis = 0: i32} : tensor<8xi32> -> tensor<1x8xi32>
  tt.return
}

// -----
// CHECK-LABEL: @expand_dims_int_axis1
tt.func @expand_dims_int_axis1(%arg0: tensor<8xi32>) {
  // CHECK: tensor.expand_shape %arg0 {{\[\[0, 1]]}}
  %0 = tt.expand_dims %arg0 {axis = 1: i32} : tensor<8xi32> -> tensor<8x1xi32>
  tt.return
}

// -----
// CHECK-LABEL: @expand_dims_ptr
// CHECK-SAME: %[[ARG:.*]]: tensor<8xi64>
tt.func @expand_dims_ptr(%arg0: tensor<8x!tt.ptr<i32>>) {
  // CHECK: tensor.expand_shape %[[ARG]] {{\[\[0, 1]]}}
  %0 = tt.expand_dims %arg0 {axis = 1: i32} : tensor<8x!tt.ptr<i32>> -> tensor<8x1x!tt.ptr<i32>>
  tt.return
}

// -----
// CHECK-LABEL: @view_to_high_rank_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<256x!tt.ptr<f32>>
func.func @view_to_high_rank_ptr(%arg0: tensor<256x!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<256x!tt.ptr<f32>> to tensor<256xi64>
  // CHECK: tensor.expand_shape %[[ARG1]] {{\[\[0, 1, 2]]}}
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<256x!tt.ptr<f32>> -> tensor<8x16x2x!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @view_to_low_rank_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x8x16x!tt.ptr<f32>>
func.func @view_to_low_rank_ptr(%arg0: tensor<2x8x16x!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<2x8x16x!tt.ptr<f32>> to tensor<2x8x16xi64>
  // CHECK: tensor.collapse_shape %[[ARG1]] {{\[\[0, 1, 2]]}}
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<2x8x16x!tt.ptr<f32>> -> tensor<256x!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @view_same_rank_src_has_dim_1
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x64xf32>
func.func @view_same_rank_src_has_dim_1(%arg0: tensor<1x64xf32>) {
  // CHECK-NEXT: %[[ARG1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[0, 1]]}}
  // CHECK-NEXT: tensor.expand_shape %[[ARG1]] {{\[\[0, 1]]}}
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<1x64xf32> -> tensor<4x16xf32>
  return
}

// -----
// CHECK-LABEL: @view_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<16x16x!tt.ptr<f32>>
func.func @view_ptr(%arg0: tensor<16x16x!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<16x16x!tt.ptr<f32>> to tensor<16x16xi64>
  // CHECK: %[[ARG2:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[0, 1]]}}
  // CHECK: tensor.expand_shape %[[ARG2]] {{\[\[0, 1]]}}
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<16x16x!tt.ptr<f32>> -> tensor<4x64x!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @view_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x4x!tt.ptr<f32>>
func.func @view_ptr(%arg0: tensor<4x4x!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<4x4x!tt.ptr<f32>> to tensor<4x4xi64>
  // CHECK: %[[ARG2:.*]] = tensor.collapse_shape %[[ARG1]] {{\[\[0, 1]]}}
  // CHECK: tensor.expand_shape %[[ARG2]] {{\[\[0, 1]]}}
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<4x4x!tt.ptr<f32>> -> tensor<1x16x!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @view_to_same_shape_ptr
// CHECK-SAME: %[[ARG:.*]]: tensor<16x16x!tt.ptr<f32>>
func.func @view_to_same_shape_ptr(%arg0: tensor<16x16x!tt.ptr<f32>>) -> tensor<16x16x!tt.ptr<f32>> {
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<16x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
  // CHECK: return %[[ARG]] : tensor<16x16x!tt.ptr<f32>>
  return %0 : tensor<16x16x!tt.ptr<f32>>
}

// -----
// CHECK-LABEL: @view_0d_1d_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<!tt.ptr<f32>>
func.func @view_0d_1d_ptr(%arg0: tensor<!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<!tt.ptr<f32>> to tensor<i64>
  // CHECK: tensor.expand_shape %[[ARG1]] []
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<!tt.ptr<f32>> -> tensor<1x!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @view_1d_0d_ptr
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x!tt.ptr<f32>>
func.func @view_1d_0d_ptr(%arg0: tensor<1x!tt.ptr<f32>>) {
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<1x!tt.ptr<f32>> to tensor<1xi64>
  // CHECK: tensor.collapse_shape %[[ARG1]] []
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<1x!tt.ptr<f32>> -> tensor<!tt.ptr<f32>>
  return
}

// -----
// CHECK-LABEL: @add_ptr
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<256xi32>
tt.func @add_ptr(%arg0: !tt.ptr<f32>, %arg1: tensor<256xi32>) {
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[ARG0]] : i64) outs(%[[INIT]] : tensor<256xi64>)
  // CHECK-NEXT: %[[INIT2:.*]] = tensor.empty
  // CHECK-NEXT: %[[OUT:.*]] = linalg.map ins(%[[FILL]], %[[ARG1]] : tensor<256xi64>, tensor<256xi32>) outs(%[[INIT2]] : tensor<256xi64>)
  // CHECK-NEXT: (%[[IN1:.*]]: i64, %[[IN2:.*]]: i32)
  // CHECK-NEXT:     %[[EXT:.*]] = arith.extsi %[[IN2]] : i32 to i64
  // CHECK-NEXT:     %[[C4:.*]] = arith.constant 4
  // CHECK-NEXT:     %[[MUL:.*]] = arith.muli %[[EXT]], %[[C4]]
  // CHECK-NEXT:     %[[ADD:.*]] = arith.addi %[[IN1]], %[[MUL]]
  // CHECK-NEXT:     linalg.yield %[[ADD]]
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1: tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  tt.return
}

// -----
// CHECK-LABEL: @add_ptr_for_scalar
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32
tt.func @add_ptr_for_scalar(%arg0: !tt.ptr<f32>, %arg1: i32) {
  // CHECK-NEXT: %[[EXT:.*]] = arith.extsi %[[ARG1]] : i32 to i64
  // CHECK-NEXT: %[[C4:.*]] = arith.constant 4
  // CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[EXT]], %[[C4]]
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[MUL]]
  %0 = tt.addptr %arg0, %arg1: !tt.ptr<f32>, i32
  tt.return
}

// -----
// Offset of tt.addptr is i64 which is the same type as ptr as int.
// CHECK-LABEL: @add_ptr_for_scalar_i64
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64
tt.func @add_ptr_for_scalar_i64(%arg0: !tt.ptr<f32>, %arg1: i64) {
  // CHECK-NEXT: %[[C4:.*]] = arith.constant 4
  // CHECK-NEXT: %[[MUL:.*]] = arith.muli %[[ARG1]], %[[C4]]
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[MUL]]
  %0 = tt.addptr %arg0, %arg1: !tt.ptr<f32>, i64
  tt.return
}

// -----
// CHECK-LABEL: @make_range
tt.func @make_range() {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[START:.*]] = arith.constant 1 : i32
  // CHECK-NEXT: %[[END:.*]] = arith.constant 129 : i32
  // CHECK-NEXT: linalg_ext.make_range {operandSegmentSizes = array<i32: 2, 1>} ins(%[[START]], %[[END]] : i32, i32) outs(%[[INIT]] : tensor<128xi32>)
  %0 = tt.make_range {end = 129 : i32, start = 1 : i32} : tensor<128xi32>
  tt.return
}

// -----
// CHECK-LABEL: @dot
// CHECK-SAME:    %[[ARG0:.*]]: tensor<64x64xf16>, %[[ARG1:.*]]: tensor<64x64xf16>, %[[ARG2:.*]]: tensor<64x64xf32>
tt.func @dot(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf32>) {
  // CHECK: %[[MATMUL:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<64x64xf16>, tensor<64x64xf16>) outs(%[[ARG2]] : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
  tt.return
}

// -----
// CHECK-LABEL: @dot_allow_tf32_false
// CHECK-SAME:    %[[ARG0:.*]]: tensor<64x64xf16>, %[[ARG1:.*]]: tensor<64x64xf16>, %[[ARG2:.*]]: tensor<64x64xf32>
tt.func @dot_allow_tf32_false(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf32>) {
  // CHECK: %[[MATMUL:.*]] = linalg.matmul {__allow_tf32__} ins(%[[ARG0]], %[[ARG1]] : tensor<64x64xf16>, tensor<64x64xf16>) outs(%[[ARG2]] : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
  tt.return
}

// -----
// CHECK-LABEL: @bitcast_scalar
tt.func @bitcast_scalar(%arg0: i32) {
  // CHECK: arith.bitcast %arg0 : i32 to f32
  %0 = tt.bitcast %arg0 : i32 -> f32
  tt.return
}

// -----
// CHECK-LABEL: @bitcast_scalar_ptr
// CHECK-SAME: %[[ARG:.*]]: i64
tt.func @bitcast_scalar_ptr(%arg0: !tt.ptr<f32>) {
  // CHECK: arith.bitcast %[[ARG]] : i64 to i64
  %0 = tt.bitcast %arg0 : !tt.ptr<f32> -> i64
  tt.return
}

// -----
// CHECK-LABEL: @bitcast
// CHECK-SAME: %[[ARG:.*]]: tensor<128xi32>
tt.func @bitcast(%arg0: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: linalg.map { arith.bitcast } ins(%[[ARG]] : tensor<128xi32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = tt.bitcast %arg0 : tensor<128xi32> -> tensor<128xf32>
  tt.return
}

// -----
// CHECK-LABEL: @bitcast_ptr
// CHECK-SAME: %[[ARG:.*]]: tensor<128xi64>
tt.func @bitcast_ptr(%arg0: tensor<128x!tt.ptr<f32>>) {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: linalg.map { arith.bitcast } ins(%[[ARG]] : tensor<128xi64>) outs(%[[INIT]] : tensor<128xi64>)
  %0 = tt.bitcast %arg0 : tensor<128x!tt.ptr<f32>> -> tensor<128xi64>
  tt.return
}

// -----
tt.func @reduce_min_2d_f16(%arg0: tensor<1x2048xf16>) {
  // CHECK-LABEL:   func.func @reduce_min_2d_f16(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf16>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0x7C00 : f16
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf16>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f16) outs(%[[VAL_2]] : tensor<1xf16>) -> tensor<1xf16>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf16>) outs(%[[VAL_3]] : tensor<1xf16>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16) {
  // CHECK:               %[[VAL_7:.*]] = arith.minimumf %[[VAL_5]], %[[VAL_6]] : f16
  // CHECK:               linalg.yield %[[VAL_7]] : f16
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f16, %arg2: f16):
    %1 = arith.minimumf %arg1, %arg2 : f16
    tt.reduce.return %1 : f16
  }) {axis = 1 : i32} : (tensor<1x2048xf16>) -> tensor<1xf16>
  tt.return
}

// -----
tt.func @reduce_min_2d_f32(%arg0: tensor<1x2048xf32>) {
  // CHECK-LABEL:   func.func @reduce_min_2d_f32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0x7F800000 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf32>) outs(%[[VAL_3]] : tensor<1xf32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.minimumf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.minimumf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 1 : i32} : (tensor<1x2048xf32>) -> tensor<1xf32>
  tt.return
}

// -----
tt.func @reduce_min_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_min_2d_i32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 2147483647 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.minsi %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.minsi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_min_2d_ui32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_min_2d_ui32(
  // CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant -1 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.minui %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.minui %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_max_2d_f16(%arg0: tensor<1x2048xf16>) {
  // CHECK-LABEL:   func.func @reduce_max_2d_f16(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf16>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0xFC00 : f16
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf16>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f16) outs(%[[VAL_2]] : tensor<1xf16>) -> tensor<1xf16>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf16>) outs(%[[VAL_3]] : tensor<1xf16>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16) {
  // CHECK:               %[[VAL_7:.*]] = arith.maximumf %[[VAL_5]], %[[VAL_6]] : f16
  // CHECK:               linalg.yield %[[VAL_7]] : f16
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f16, %arg2: f16):
    %1 = arith.maximumf %arg1, %arg2 : f16
    tt.reduce.return %1 : f16
  }) {axis = 1 : i32} : (tensor<1x2048xf16>) -> tensor<1xf16>
  tt.return
}

// -----
tt.func @reduce_max_2d_f32(%arg0: tensor<1x2048xf32>) {
  // CHECK-LABEL:   func.func @reduce_max_2d_f32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0xFF800000 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf32>) outs(%[[VAL_3]] : tensor<1xf32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.maximumf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.maximumf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 1 : i32} : (tensor<1x2048xf32>) -> tensor<1xf32>
  tt.return
}

// -----
tt.func @reduce_max_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_max_2d_i32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant -2147483648 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.maxsi %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.maxsi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_max_2d_ui32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_max_2d_ui32(
  // CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.maxui %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.maxui %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_sum_2d_f32(%arg0: tensor<1x2048xf32>) {
  // CHECK-LABEL:   func.func @reduce_sum_2d_f32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<1xf32>) -> tensor<1xf32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf32>) outs(%[[VAL_3]] : tensor<1xf32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 1 : i32} : (tensor<1x2048xf32>) -> tensor<1xf32>
  tt.return
}

// -----
tt.func @reduce_sum_2d_f32(%arg0: tensor<1x2048xf32>) {
  // CHECK-LABEL:   func.func @reduce_sum_2d_f32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xf32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2048xf32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<2048xf32>) -> tensor<2048xf32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xf32>) outs(%[[VAL_3]] : tensor<2048xf32>) dimensions = [0]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 0 : i32} : (tensor<1x2048xf32>) -> tensor<2048xf32>
  tt.return
}

// -----
tt.func @reduce_sum_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_sum_2d_i32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_xor_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_xor_2d_i32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1x2048xi32>) outs(%[[VAL_3]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.xori %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.xori %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_addf_1d(%arg0: tensor<4xf32>) {
  // CHECK-LABEL:   func.func @reduce_addf_1d(
  // CHECK-SAME:                              %[[VAL_0:.*]]: tensor<4xf32>) {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<f32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<f32>) -> tensor<f32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<4xf32>) outs(%[[VAL_3]] : tensor<f32>) dimensions = [0]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           %[[VAL_8:.*]] = tensor.extract %[[VAL_4]][] : tensor<f32>
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
  tt.return
}

// -----
tt.func @reduce_addf_1d_size1(%arg0: tensor<1xf32>) -> f32 {
  // CHECK-LABEL:   func.func @reduce_addf_1d_size1(
  // CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<1xf32>) -> f32 {
  // CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<f32>
  // CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<f32>) -> tensor<f32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<1xf32>) outs(%[[VAL_3]] : tensor<f32>) dimensions = [0]
  // CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
  // CHECK:               %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
  // CHECK:               linalg.yield %[[VAL_7]] : f32
  // CHECK:             }
  // CHECK:           %[[VAL_8:.*]] = tensor.extract %[[VAL_4]][] : tensor<f32>
  // CHECK:           return %[[VAL_8]] : f32
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 0 : i32} : (tensor<1xf32>) -> f32
  tt.return %0 : f32
}

// -----
tt.func @reduce_subf_1d_size1(%arg0: tensor<1xf32>) -> f32 {
  // CHECK-LABEL:   func.func @reduce_subf_1d_size1(
  // CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<1xf32>) -> f32 {
  // CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] [] : tensor<1xf32> into tensor<f32>
  // CHECK:           %[[VAL_2:.*]] = tensor.extract %[[VAL_1]][] : tensor<f32>
  // CHECK:           return %[[VAL_2]] : f32
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.subf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  }) {axis = 0 : i32} : (tensor<1xf32>) -> f32
  tt.return %0 : f32
}

// -----
tt.func @reduce_sub_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_sub_2d_i32(
  // CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [1, 1] [1, 1] : tensor<1x2048xi32> to tensor<1x1xi32>
  // CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<1x1xi32> into tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, 1] [1, 2047] [1, 1] : tensor<1x2048xi32> to tensor<1x2047xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_3]] : tensor<1x2047xi32>) outs(%[[VAL_2]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i32
  // CHECK:               linalg.yield %[[VAL_7]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.subi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
tt.func @reduce_no_payload_2d_i32(%arg0: tensor<1x2048xi32>) {
  // CHECK-LABEL:   func.func @reduce_no_payload_2d_i32(
  // CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<1x2048xi32>) {
  // CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [1, 1] [1, 1] : tensor<1x2048xi32> to tensor<1x1xi32>
  // CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<1x1xi32> into tensor<1xi32>
  // CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, 1] [1, 2047] [1, 1] : tensor<1x2048xi32> to tensor<1x2047xi32>
  // CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_3]] : tensor<1x2047xi32>) outs(%[[VAL_2]] : tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
  // CHECK:               linalg.yield %[[VAL_6]] : i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %0 = "tt.reduce" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    tt.reduce.return %arg2 : i32
  }) {axis = 1 : i32} : (tensor<1x2048xi32>) -> tensor<1xi32>
  tt.return
}

// -----
// CHECK-LABEL: @reduce_mix_3d_f32
tt.func @reduce_mix_3d_f32(%arg0: tensor<1x2048x32xf32>, %arg1: tensor<1x2048x32xf32>) {
  // CHECK: %[[ARG0_FIRST:.*]] = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 32] [1, 1, 1] : tensor<1x2048x32xf32> to tensor<1x1x32xf32>
  // CHECK-NEXT: %[[ARG0_COLLAPSE:.*]] = tensor.collapse_shape %[[ARG0_FIRST]] {{\[}}[0, 1], [2]] : tensor<1x1x32xf32> into tensor<1x32xf32>
  // CHECK-NEXT: %[[ARG0_SLICE:.*]] = tensor.extract_slice %arg0[0, 1, 0] [1, 2047, 32] [1, 1, 1] : tensor<1x2048x32xf32> to tensor<1x2047x32xf32>
  // CHECK-NEXT: %[[ARG1_FIRST:.*]] = tensor.extract_slice %arg1[0, 0, 0] [1, 1, 32] [1, 1, 1] : tensor<1x2048x32xf32> to tensor<1x1x32xf32>
  // CHECK-NEXT: %[[ARG1_COLLAPSE:.*]] = tensor.collapse_shape %[[ARG1_FIRST]] {{\[}}[0, 1], [2]] : tensor<1x1x32xf32> into tensor<1x32xf32>
  // CHECK-NEXT: %[[ARG1_SLICE:.*]] = tensor.extract_slice %arg1[0, 1, 0] [1, 2047, 32] [1, 1, 1] : tensor<1x2048x32xf32> to tensor<1x2047x32xf32>
  // CHECK-NEXT: %[[REDUCE:.*]]:2 = linalg.reduce ins(%[[ARG0_SLICE]], %[[ARG1_SLICE]] : tensor<1x2047x32xf32>, tensor<1x2047x32xf32>) outs(%[[ARG0_COLLAPSE]], %[[ARG1_COLLAPSE]] : tensor<1x32xf32>, tensor<1x32xf32>) dimensions = [1]
  // CHECK-NEXT: (%[[IN1:.*]]: f32, %[[IN2:.*]]: f32, %[[INIT1:.*]]: f32, %[[INIT2:.*]]: f32) {
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[IN1]], %[[INIT1]] : f32
  // CHECK-NEXT: %[[MAX:.*]] = arith.maximumf %[[IN2]], %[[INIT2]] : f32
  // CHECK-NEXT: linalg.yield %[[ADD]], %[[MAX]] : f32, f32
  %0, %1 = "tt.reduce" (%arg0, %arg1) ({
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg2, %arg4 : f32
    %3 = arith.maximumf %arg3, %arg5 : f32
    tt.reduce.return %2, %3 : f32, f32
  }) {axis = 1 : i32} : (tensor<1x2048x32xf32>, tensor<1x2048x32xf32>) -> (tensor<1x32xf32>, tensor<1x32xf32>)
  tt.return
}

// -----
tt.func @reduce_multi_statement_argmin_f32(%arg0: tensor<1x256xf32>, %arg1: tensor<1x256xi32>) {
  // CHECK-LABEL:   func.func @reduce_multi_statement_argmin_f32(
  // CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<1x256xf32>,
  // CHECK-SAME:                                                 %[[VAL_1:.*]]: tensor<1x256xi32>) {
  // CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [1, 1] [1, 1] : tensor<1x256xf32> to tensor<1x1xf32>
  // CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  // CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0, 1] [1, 255] [1, 1] : tensor<1x256xf32> to tensor<1x255xf32>
  // CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][0, 0] [1, 1] [1, 1] : tensor<1x256xi32> to tensor<1x1xi32>
  // CHECK:           %[[VAL_6:.*]] = tensor.collapse_shape %[[VAL_5]] {{\[\[}}0, 1]] : tensor<1x1xi32> into tensor<1xi32>
  // CHECK:           %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_1]][0, 1] [1, 255] [1, 1] : tensor<1x256xi32> to tensor<1x255xi32>
  // CHECK:           %[[VAL_8:.*]]:2 = linalg.reduce ins(%[[VAL_4]], %[[VAL_7]] : tensor<1x255xf32>, tensor<1x255xi32>) outs(%[[VAL_3]], %[[VAL_6]] : tensor<1xf32>, tensor<1xi32>) dimensions = [1]
  // CHECK:             (%[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: i32) {
  // CHECK:               %[[VAL_13:.*]] = arith.cmpf olt, %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               %[[VAL_14:.*]] = arith.cmpf ogt, %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_12]] : i32
  // CHECK:               %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_10]], %[[VAL_12]] : i32
  // CHECK:               %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_16]] : i32
  // CHECK:               %[[VAL_18:.*]] = arith.select %[[VAL_13]], %[[VAL_10]], %[[VAL_17]] : i32
  // CHECK:               %[[VAL_19:.*]] = arith.cmpf olt, %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               linalg.yield %[[VAL_20]], %[[VAL_18]] : f32, i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %9:2 = "tt.reduce"(%arg0, %arg1) ({
  ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
    %14 = arith.cmpf olt, %arg4, %arg6 : f32
    %15 = arith.cmpf ogt, %arg4, %arg6 : f32
    %16 = arith.cmpi slt, %arg5, %arg7 : i32
    %17 = arith.select %16, %arg5, %arg7 : i32
    %18 = arith.select %15, %arg7, %17 : i32
    %19 = arith.select %14, %arg5, %18 : i32
    %20 = arith.cmpf olt, %arg4, %arg6 : f32
    %21 = arith.select %20, %arg4, %arg6 : f32
    tt.reduce.return %21, %19 : f32, i32
  }) {axis = 1 : i32} : (tensor<1x256xf32>, tensor<1x256xi32>) -> (tensor<1xf32>, tensor<1xi32>)
  tt.return
}

// -----
tt.func @reduce_multi_statement_argmax_f32(%arg0: tensor<2x2x256xf32>, %arg1: tensor<2x2x256xi32>) {
  // CHECK-LABEL:   func.func @reduce_multi_statement_argmax_f32(
  // CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<2x2x256xf32>,
  // CHECK-SAME:                                                 %[[VAL_1:.*]]: tensor<2x2x256xi32>) {
  // CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] [2, 2, 1] [1, 1, 1] : tensor<2x2x256xf32> to tensor<2x2x1xf32>
  // CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : tensor<2x2x1xf32> into tensor<2x2xf32>
  // CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1] [2, 2, 255] [1, 1, 1] : tensor<2x2x256xf32> to tensor<2x2x255xf32>
  // CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][0, 0, 0] [2, 2, 1] [1, 1, 1] : tensor<2x2x256xi32> to tensor<2x2x1xi32>
  // CHECK:           %[[VAL_6:.*]] = tensor.collapse_shape %[[VAL_5]] {{\[\[}}0], [1, 2]] : tensor<2x2x1xi32> into tensor<2x2xi32>
  // CHECK:           %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_1]][0, 0, 1] [2, 2, 255] [1, 1, 1] : tensor<2x2x256xi32> to tensor<2x2x255xi32>
  // CHECK:           %[[VAL_8:.*]]:2 = linalg.reduce ins(%[[VAL_4]], %[[VAL_7]] : tensor<2x2x255xf32>, tensor<2x2x255xi32>) outs(%[[VAL_3]], %[[VAL_6]] : tensor<2x2xf32>, tensor<2x2xi32>) dimensions = [2]
  // CHECK:             (%[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: i32) {
  // CHECK:               %[[VAL_13:.*]] = arith.cmpf ogt, %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_9]], %[[VAL_11]] : f32
  // CHECK:               %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_10]], %[[VAL_12]] : i32
  // CHECK:               linalg.yield %[[VAL_14]], %[[VAL_15]] : f32, i32
  // CHECK:             }
  // CHECK:           return
  // CHECK:         }
  %9:2 = "tt.reduce"(%arg0, %arg1) ({
  ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
    %1 = arith.cmpf ogt, %arg4, %arg6 : f32
    %2 = arith.select %1, %arg4, %arg6 : f32
    %3 = arith.select %1, %arg5, %arg7 : i32
    tt.reduce.return %2, %3 : f32, i32
  }) {axis = 2 : i32} : (tensor<2x2x256xf32>, tensor<2x2x256xi32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  tt.return
}

// -----
// CHECK-LABEL: @for_iter_args
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128x64xi32>
tt.func @for_iter_args(%arg0: !tt.ptr<f16>, %arg1: tensor<128x64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128x64xi64>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ARG0]] : i64) outs(%[[INIT]] : tensor<128x64xi64>) -> tensor<128x64xi64>
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  // CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %c0{{.*}} to %c10{{.*}} step %c1{{.*}} iter_args(%[[IA:.*]] = %[[FILL]]) -> (tensor<128x64xi64>)
  %1 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %0) -> (tensor<128x64x!tt.ptr<f16>>) {
    // CHECK: %[[ADD:.*]] = linalg.map
    %2 = tt.addptr %arg3, %arg1 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    // CHECK: scf.yield %[[ADD]] : tensor<128x64xi64>
    scf.yield %2 : tensor<128x64x!tt.ptr<f16>>
  }

  tt.return
}

// -----
// CHECK-LABEL: @cast_ptr_and_int_scalar
tt.func @cast_ptr_and_int_scalar(%arg0: !tt.ptr<f32>) {
  // CHECK-NEXT: return
  %0 = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
  %1 = tt.int_to_ptr %0 : i64 -> !tt.ptr<f32>
  tt.return
}

// -----
// CHECK-LABEL: @cast_ptr_and_int_1D
tt.func @cast_ptr_and_int_1D(%arg0: tensor<16x!tt.ptr<f32>>) {
  // CHECK-NEXT: return
  %0 = tt.ptr_to_int %arg0 : tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
  %1 = tt.int_to_ptr %0 : tensor<16xi64> -> tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @cast_ptr_and_int_2D
tt.func @cast_ptr_and_int_2D(%arg0: tensor<2x16x!tt.ptr<f32>>) {
  // CHECK-NEXT: return
  %0 = tt.ptr_to_int %arg0 : tensor<2x16x!tt.ptr<f32>> -> tensor<2x16xi64>
  %1 = tt.int_to_ptr %0 : tensor<2x16xi64> -> tensor<2x16x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @cast_ptr_and_int_2D_tt.return_used
tt.func @cast_ptr_and_int_2D_tt.return_used(%scalar_i64: i64) -> tensor<2x16xi64> {
  // CHECK: tensor.empty() : tensor<2x16xi64>
  // CHECK: linalg.fill ins(%arg0 : i64) outs(%0 : tensor<2x16xi64>) -> tensor<2x16xi64>
  %tensor_i64_1d = tt.splat %scalar_i64 : i64 -> tensor<2x16xi64>
  // CHECK-NEXT: return
  %0 = tt.int_to_ptr %tensor_i64_1d : tensor<2x16xi64> -> tensor<2x16x!tt.ptr<f32>>
  %1 = tt.ptr_to_int %0 : tensor<2x16x!tt.ptr<f32>> -> tensor<2x16xi64>
  tt.return %1 : tensor<2x16xi64>
}

// -----
// CHECK-LABEL: @optimize_barrier
// CHECK-SAME: %[[ARG:.*]]: i64
tt.func @optimize_barrier(%arg0: !tt.ptr<f32>) {
  // CHECK: aux.optimization_barrier %[[ARG]] : i64
  aux.optimization_barrier %arg0 : !tt.ptr<f32>
  tt.return
}

// -----
// CHECK-LABEL: @trans_2d
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16x32xf32>
tt.func @trans_2d(%arg0: tensor<16x32xf32>) {
  // CHECK: %[[INIT_OUT:.*]] = tensor.empty() : tensor<32x16xf32>
  // CHECK: %[[TRANSPOSED_OUT:.*]] = linalg.transpose ins(%[[ARG0]] : tensor<16x32xf32>) outs(%[[INIT_OUT]] : tensor<32x16xf32>) permutation = [1, 0]
  %out = tt.trans %arg0 {order=array<i32: 1, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
  tt.return
}

// -----
// CHECK-LABEL: @trans_1d
// CHECK-SAME:  %[[ARG0:.*]]: tensor<16xf32>
tt.func @trans_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK-NOT: tt.trans
  %out = tt.trans %arg0 {order=array<i32:0>} : tensor<16xf32> -> tensor<16xf32>
  // CHECK: return %[[ARG0]]
  tt.return %out : tensor<16xf32>
}

// -----
// CHECK-LABEL: @trans_0d
// CHECK-SAME:  %[[ARG0:.*]]: tensor<f32>
tt.func @trans_0d(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: tt.trans
  %out = tt.trans %arg0 {order=array<i32>} : tensor<f32> -> tensor<f32>
  // CHECK: return %[[ARG0]]
  tt.return %out : tensor<f32>
}

// -----
// CHECK-LABEL: @cmpi_to_fill
func.func @cmpi_to_fill(%arg0: i32) {
  // CHECK: %[[ARG:.*]] = arith.index_cast %arg0 : i32 to index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[ARG1:.*]] = arith.maxsi %[[ARG]], %[[C0]] : index
  // CHECK: %[[C128:.*]] = arith.constant 128 : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[C128]], %[[ARG1]]
  // CHECK: %[[INIT1:.*]] = tensor.empty(%[[SIZE]])
  // CHECK: %[[TRUE:.*]] = linalg.fill ins(%true : i1) outs(%[[INIT1]] : tensor<?xi1>)
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: %[[PAD_INIT:.*]] = tensor.empty() : tensor<128xi1>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[PAD_INIT]], %[[C0_0]] : tensor<128xi1>
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[ADD:.*]] = arith.addi %[[C0_1]], %[[SIZE]] : index
  // CHECK: %[[SUB:.*]] = arith.subi %[[DIM]], %[[ADD]] : index
  // CHECK: linalg_ext.pad ins(%[[TRUE]] : tensor<?xi1>) outs(%[[PAD_INIT]] : tensor<128xi1>) pvalue(%[[FALSE]] : i1) low = [%[[C0_1]]] high = [%[[SUB]]]
  // CHECK-NOT: arith.cmpi
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg0 : i32 -> tensor<128xi32>
  %mask = arith.cmpi slt, %0, %1 : tensor<128xi32>
  return
}

// -----
// CHECK-LABEL: func.func @cmp2fill_ub
// CHECK-SAME: (%[[ARG_I32:.*]]: i32)
// CHECK: %[[ARG:.*]] = arith.index_cast %[[ARG_I32]] : i32 to index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ARG1:.*]] = arith.addi %[[ARG]], %[[C1]] : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[LB0:.*]] = arith.minsi %[[C128]], %[[ARG1]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LB1:.*]] = arith.maxsi %[[C0]], %[[LB0]] : index
// CHECK: %[[C128_0:.*]] = arith.constant 128 : index
// CHECK: %[[SIZE:.*]] = arith.subi %[[C128_0]], %[[LB1]] : index
// CHECK: %[[INIT1:.*]] = tensor.empty(%[[SIZE]]) : tensor<?xi1>
// CHECK: %[[C_TRUE:.*]] = arith.constant true
// CHECK: %[[TRUE:.*]] = linalg.fill ins(%[[C_TRUE]] : i1) outs(%[[INIT1]] : tensor<?xi1>)
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[PAD_INIT:.*]] = tensor.empty() : tensor<128xi1>
// CHECK: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM:.*]] = tensor.dim %[[PAD_INIT]], %[[C0_0]] : tensor<128xi1>
// CHECK: %[[C_DIM:.*]] = arith.addi %[[LB1]], %[[SIZE]] : index
// CHECK: %[[C_0:.*]] = arith.subi %[[DIM]], %[[C_DIM]] : index
// CHECK: linalg_ext.pad ins(%[[TRUE]] : tensor<?xi1>) outs(%[[PAD_INIT]] : tensor<128xi1>) pvalue(%[[FALSE]] : i1) low = [%[[LB1]]] high = [%[[C_0]]] {
// CHECK-NOT: arith.cmpi
func.func @cmp2fill_ub(%arg: i32) -> tensor<128xi1> {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %arg : i32 -> tensor<128xi32>
  %2 = arith.cmpi sgt, %0, %1 : tensor<128xi32>
  return %2 : tensor<128xi1>
}



// -----
// CHECK-LABEL: @select_conversion
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<64x64xf32>
func.func @select_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i32, %arg3: index) -> tensor<64x64xf32> {
    %cst = arith.constant dense<50304> : tensor<64x1xi32>
    %cst_0 = arith.constant dense<50304> : tensor<1x64xi32>
    %cst_1 = arith.constant dense<2048> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg2, %c64_i32 : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = tt.splat %0 : i32 -> tensor<64x1xi32>
    %4 = arith.addi %3, %2 : tensor<64x1xi32>
    %5 = arith.cmpi slt, %4, %cst_1 : tensor<64x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %5 : tensor<64x1xi1> -> tensor<64x64xi1>
    %8 = arith.index_cast %arg3 : index to i32
    %9 = tt.splat %8 : i32 -> tensor<1x64xi32>
    %10 = arith.addi %9, %6 : tensor<1x64xi32>
    %11 = arith.cmpi slt, %10, %cst_0 : tensor<1x64xi32>
    %12 = tt.broadcast %11 : tensor<1x64xi1> -> tensor<64x64xi1>
    %13 = arith.andi %12, %7 : tensor<64x64xi1>
    // CHECK: %[[EXTRACTSLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[OFFSET0:.*]], %[[OFFSET1:.*]]] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    // CHECK-NEXT: tensor.insert_slice %[[EXTRACTSLICE]] into %[[ARG1]][0, 0] [%[[OFFSET0]], %[[OFFSET1]]] [1, 1] : tensor<?x?xf32> into tensor<64x64xf32>
    // CHECK-NOT: arith.select
    %14 = arith.select %13, %arg0, %arg1 : tensor<64x64xi1>, tensor<64x64xf32>
    return %14 : tensor<64x64xf32>
}

// -----
// CHECK-LABEL: @select_decompose_conversion
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<64x64xf32>
func.func @select_decompose_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i32, %arg3: index) -> tensor<64x64xf32> {
    %cst = arith.constant dense<50304> : tensor<64x1xi32>
    %cst_0 = arith.constant dense<50304> : tensor<1x64xi32>
    %cst_1 = arith.constant dense<2048> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg2, %c64_i32 : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = tt.splat %0 : i32 -> tensor<64x1xi32>
    %4 = arith.addi %3, %2 : tensor<64x1xi32>
    %5 = arith.cmpi slt, %4, %cst_1 : tensor<64x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %5 : tensor<64x1xi1> -> tensor<64x64xi1>
    %8 = arith.index_cast %arg3 : index to i32
    %9 = tt.splat %8 : i32 -> tensor<1x64xi32>
    %10 = arith.addi %9, %6 : tensor<1x64xi32>
    %11 = arith.cmpi slt, %10, %cst_0 : tensor<1x64xi32>
    %12 = tt.broadcast %11 : tensor<1x64xi1> -> tensor<64x64xi1>
    %13 = arith.andi %12, %7 : tensor<64x64xi1>
    %14 = arith.cmpf olt, %arg1, %arg0 : tensor<64x64xf32>
    %15 = arith.andi %13, %14 : tensor<64x64xi1>
    // CHECK: %[[CMP:.*]] = linalg.map { arith.cmpf {predicate = 4 : i64} } ins(%[[ARG1]], %[[ARG0]] : tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<64x64xf32>
    // CHECK-NEXT: %[[SELECT:.*]] = linalg.map { arith.select } ins(%[[CMP]], %[[ARG0]], %[[ARG1]] : tensor<64x64xi1>, tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[INIT]] : tensor<64x64xf32>)
    // CHECK-NEXT: %[[EXTRACTSLICE:.*]] = tensor.extract_slice %[[SELECT]][0, 0] [%[[OFFSET0:.*]], %[[OFFSET1:.*]]] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    // CHECK-NEXT: tensor.insert_slice %[[EXTRACTSLICE]] into %[[ARG1]][0, 0] [%[[OFFSET0]], %[[OFFSET1]]] [1, 1] : tensor<?x?xf32> into tensor<64x64xf32>
    %16 = arith.select %15, %arg0, %arg1 : tensor<64x64xi1>, tensor<64x64xf32>
    return %16 : tensor<64x64xf32>
}

// -----
// CHECK-LABEL: @select_pad_conversion
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<64x64xf32>
func.func @select_pad_conversion(%arg0: tensor<64x64xf32>, %arg1: i32) -> tensor<64x64xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
  %cst_0 = arith.constant dense<2048> : tensor<64x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %0 = arith.muli %arg1, %c64_i32 : i32
  %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %3 = tt.splat %0 : i32 -> tensor<64x1xi32>
  %4 = arith.addi %3, %2 : tensor<64x1xi32>
  %5 = arith.cmpi slt, %4, %cst_0 : tensor<64x1xi32>
  %6 = tt.broadcast %5 : tensor<64x1xi1> -> tensor<64x64xi1>
  // CHECK:      %[[EXTRACTSLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [%[[OFFSET0:.*]], 64] [1, 1] : tensor<64x64xf32> to tensor<?x64xf32>
  // CHECK-NEXT: %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT: %[[PAD_INIT:.*]] = tensor.empty() : tensor<64x64xf32>
  // CHECK-NEXT: %[[CST1_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DIM0:.*]] = tensor.dim %[[PAD_INIT]], %[[CST1_0]] : tensor<64x64xf32>
  // CHECK-NEXT: %[[CST1_1:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ADD0:.*]] = arith.addi %[[CST1_1]], %[[OFFSET0]] : index
  // CHECK-NEXT: %[[PAD_SIZE0:.*]] = arith.subi %[[DIM0]], %[[ADD0]] : index
  // CHECK-NEXT: %[[CST2:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[DIM1:.*]] = tensor.dim %[[PAD_INIT]], %[[CST2]] : tensor<64x64xf32>
  // CHECK-NEXT: %[[CST1_2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[CST3:.*]] = arith.constant 64 : index
  // CHECK-NEXT: %[[ADD1:.*]] = arith.addi %[[CST1_2]], %[[CST3]] : index
  // CHECK-NEXT: %[[PAD_SIZE1:.*]] = arith.subi %[[DIM1]], %[[ADD1]] : index
  // CHECK-NEXT: linalg_ext.pad ins(%[[EXTRACTSLICE]] : tensor<?x64xf32>) outs(%[[PAD_INIT]] : tensor<64x64xf32>) pvalue(%[[CST0]] : f32) low = [%[[CST1_1]], %[[CST1_2]]] high = [%[[PAD_SIZE0]], %[[PAD_SIZE1]]]
  %7 = arith.select %6, %arg0, %cst : tensor<64x64xi1>, tensor<64x64xf32>
  return %7 : tensor<64x64xf32>
}

// -----
// CHECK-LABEL: func.func @select_simplify_ub
// CHECK-SAME: (%[[ARG0]]: tensor<128xi32>, %[[ARG1]]: tensor<128xi32>)
// CHECK: linalg_ext.pad
// CHECK: %[[C32_INDEX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C33:.*]] = arith.addi %[[C32_INDEX]], %[[C1]] : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C33_1:.*]] = arith.minsi %[[C128]], %[[C33]] : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C33_2:.*]] = arith.maxsi %[[C0]], %[[C33_1]] : index
// CHECK: %[[C128_1:.*]] = arith.constant 128 : index
// CHECK: %[[C95:.*]] = arith.subi %[[C128_1]], %[[C33_2]] : index
// CHECK: %[[C128_2:.*]] = arith.addi %[[C33_2]], %[[C95]] : index
// CHECK: %[[EXTRACTED:.*]] = tensor.extract_slice %[[ARG0]][%[[C33_2]]] [%[[C95]]] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK: tensor.insert_slice %[[EXTRACTED]] into %[[ARG1]][%[[C33_2]]] [%[[C95]]] [1] : tensor<?xi32> into tensor<128xi32>
func.func @select_simplify_ub(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> tensor<128xi32> {
  %c32 = arith.constant 32 : i32
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %c32 : i32 -> tensor<128xi32>
  %2 = arith.cmpi sgt, %0, %1 : tensor<128xi32>
  %3 = arith.select %2, %arg0, %arg1 : tensor<128xi1>, tensor<128xi32>
  return %3 : tensor<128xi32>
}

// -----
// CHECK-LABEL: func.func @select_scalar_ptr
func.func @select_scalar_ptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  // CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>)
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to i64
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to i64
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[SCT:.*]] = arith.constant 0 : i32
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[C1]], %[[C0]] : i32
  // CHECK: %[[EXTSI:.*]] = arith.extsi %[[SCT]] : i32 to i64
  // CHECK: %[[C2:.*]] = arith.constant 4 : i64
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXTSI]], %[[C2]] : i64
  // CHECK: %[[ADD:.*]] = arith.addi %[[CAST0]], %[[MUL]] : i64
  // CHECK: %[[EXTSI1:.*]] = arith.extsi %[[SCT]] : i32 to i64
  // CHECK: %[[C3:.*]] = arith.constant 4 : i64
  // CHECK: %[[MUL1:.*]] = arith.muli %[[EXTSI1]], %[[C3]] : i64
  // CHECK: %[[ADD1:.*]] = arith.addi %[[CAST1]], %[[MUL1]] : i64
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ADD]], %[[ADD1]] : i64
  // CHECK: return
  %sct_0 = arith.constant 0 : i32
  %sct_1 = arith.constant 1 : i32
  %cst = arith.constant 0 : i32
  %0 = arith.cmpi slt, %sct_1, %sct_0 : i32
  %1 = tt.addptr %arg0, %cst : !tt.ptr<f32>, i32
  %2 = tt.addptr %arg1, %cst : !tt.ptr<f32>, i32
  %3 = arith.select %0, %1, %2 : !tt.ptr<f32>
  return
}

// -----
// CHECK-LABEL: func.func @select_tensor_ptr
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xi8>, %[[ARG1:.*]]: tensor<1024x!tt.ptr<i32>>,  %[[ARG2:.*]]: tensor<1024x!tt.ptr<i32>>)
func.func @select_tensor_ptr(%arg0: tensor<1024xi8>, %arg1: tensor<1024x!tt.ptr<i32>>, %arg2: tensor<1024x!tt.ptr<i32>>) {
    // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<1024x!tt.ptr<i32>> to tensor<1024xi64>
    // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : tensor<1024x!tt.ptr<i32>> to tensor<1024xi64>
    // CHECK: %[[CST:.*]] = arith.constant 0 : i8
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1024xi8>
    // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : i8) outs(%[[INIT]] : tensor<1024xi8>) -> tensor<1024xi8>
    // CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<1024xi1>
    // CHECK: %[[MAP:.*]] = linalg.map { arith.cmpi {predicate = 1 : i64} } ins(%[[ARG0]], %[[FILL]] : tensor<1024xi8>, tensor<1024xi8>) outs(%[[INIT1]] : tensor<1024xi1>)
    // CHECK: %[[INIT2:.*]] = tensor.empty() : tensor<1024xi64>
    // CHECK: %[[MAP2:.*]] = linalg.map { arith.select } ins(%[[MAP]], %[[CAST0]], %[[CAST1]] : tensor<1024xi1>, tensor<1024xi64>, tensor<1024xi64>) outs(%[[INIT2]] : tensor<1024xi64>)
    %cst = arith.constant dense<0> : tensor<1024xi8>
    %19 = arith.cmpi ne, %arg0, %cst : tensor<1024xi8>
    %20 = arith.select %19, %arg1, %arg2 : tensor<1024xi1>, tensor<1024x!tt.ptr<i32>>
    return
}

// -----
// CHECK-LABEL: func.func @select_decompose_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<64x64xf32>, %[[ARG2:.*]]: tensor<64x64x!tt.ptr<i32>>,  %[[ARG3:.*]]: tensor<64x64x!tt.ptr<i32>>)
func.func @select_decompose_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64x!tt.ptr<i32>>, %arg3: tensor<64x64x!tt.ptr<i32>>) {
    // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : tensor<64x64x!tt.ptr<i32>> to tensor<64x64xi64>
    // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ARG3]] : tensor<64x64x!tt.ptr<i32>> to tensor<64x64xi64>
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<64x64xi1>
    // CHECK: %[[MAP:.*]] = linalg.map { arith.cmpf {predicate = 4 : i64} } ins(%[[ARG1]], %[[ARG0]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[INIT]] : tensor<64x64xi1>)
    // CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<64x64xi64>
    // CHECK: %[[MAP1:.*]] = linalg.map { arith.select } ins(%mapped, %[[CAST0]], %[[CAST1]] : tensor<64x64xi1>, tensor<64x64xi64>, tensor<64x64xi64>) outs(%[[INIT1]] : tensor<64x64xi64>)
    %0 = arith.cmpf olt, %arg1, %arg0 : tensor<64x64xf32>
    %1 = arith.andi %0, %0 : tensor<64x64xi1>
    %2 = arith.select %1, %arg2, %arg3 : tensor<64x64xi1>, tensor<64x64x!tt.ptr<i32>>
    return
}

// -----
// CHECK-LABEL: @view_2d_to_1d
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x32xf32>
func.func @view_2d_to_1d(%arg0: tensor<1x32xf32>) {
  // CHECK: tensor.collapse_shape %[[ARG0]] {{\[\[0, 1]]}} : tensor<1x32xf32> into tensor<32xf32>
  %0 = tt.reshape %arg0 {allow_reorder = false}: tensor<1x32xf32> -> tensor<32xf32>
  return
}

// -----
func.func @gpu_barrier() {
  // CHECK-NOT: gpu.barrier
  gpu.barrier
  return
}

// -----
// CHECK-LABEL: assert_scalar_i1
// CHECK-SAME: %[[ARG0:.*]]: i32
func.func @assert_scalar_i1(%arg0: i32) {
  // CHECK: %[[CST:.*]] = arith.constant 0 : i32
  // CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[ARG0]], %[[CST]] : i32
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1xi1>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CMP]] : i1) outs(%[[INIT]] : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_scalar_i1 Assertion `lol` failed"} ins(%[[FILL]] : tensor<1xi1>) -> tensor<1xi1>
  %cst = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %cst : i32
  %1 = tt.splat %0 : i1 -> tensor<1xi1>
  tt.assert %1, "lol", "test.py", "assert_scalar_i1", 0 : tensor<1xi1>
  func.return
}

// -----
// CHECK-LABEL: assert_tensor_i1
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xi32>
func.func @assert_tensor_i1(%arg0: tensor<32xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 32 : i32
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<32xi32>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : i32) outs(%[[INIT]] : tensor<32xi32>) -> tensor<32xi32>
  // CHECK: %[[INIT2:.*]] = tensor.empty() : tensor<32xi1>
  // CHECK: %[[MAP:.*]] = linalg.map { arith.cmpi {predicate = 4 : i64} } ins(%[[ARG0]], %[[FILL]] : tensor<32xi32>, tensor<32xi32>) outs(%[[INIT2]] : tensor<32xi1>)
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_tensor_i1 Assertion `lol` failed"} ins(%[[MAP]] : tensor<32xi1>) -> tensor<32xi1>
  %cst = arith.constant dense<32> : tensor<32xi32>
  %0 = arith.cmpi sgt, %arg0, %cst : tensor<32xi32>
  tt.assert %0, "lol", "test.py", "assert_tensor_i1", 0 : tensor<32xi1>
  return
}

// -----
// CHECK-LABEL: assert_shape_one
// CHECK-SAME: %[[ARG0:.*]]: tensor<1xi32>
func.func @assert_shape_one(%arg0: tensor<1xi32>) {
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_shape_one Assertion `lol` failed"} ins(%[[ARG0]] : tensor<1xi32>) -> tensor<1xi32>
  tt.assert %arg0, "lol", "test.py", "assert_shape_one", 0 : tensor<1xi32>
  return
}

// -----
// CHECK-LABEL: assert_rank_zero
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @assert_rank_zero(%arg0: tensor<i32>) {
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_rank_zero Assertion `lol` failed"} ins(%[[ARG0]] : tensor<i32>) -> tensor<i32>
  tt.assert %arg0, "lol", "test.py", "assert_rank_zero", 0 : tensor<i32>
  return
}

// -----
// CHECK-LABEL: assert_tensor_i32
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xi32>
func.func @assert_tensor_i32(%arg0: tensor<32xi32>) {
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_tensor_i32 Assertion `lol` failed"} ins(%[[ARG0]] : tensor<32xi32>) -> tensor<32xi32>
  tt.assert %arg0, "lol", "test.py", "assert_tensor_i32", 0 : tensor<32xi32>
  return
}

// -----
// CHECK-LABEL: @assert_tensor_rank_two
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x32xi32>
func.func @assert_tensor_rank_two(%arg0: tensor<2x32xi32>) {
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_tensor_rank_two Assertion `lol` failed"} ins(%[[ARG0]] : tensor<2x32xi32>) -> tensor<2x32xi32>
  tt.assert %arg0, "lol", "test.py", "assert_tensor_rank_two", 0 : tensor<2x32xi32>
  return
}

// -----
func.func @assert_tensor_i8(%arg0: tensor<32xi8>) {
  // CHECK-LABEL: @assert_tensor_i8
  // CHECK-SAME: %[[ARG0:.*]]: tensor<32xi8>
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_tensor_i8 Assertion `lol` failed"} ins(%[[ARG0]] : tensor<32xi8>) -> tensor<32xi8>
  tt.assert %arg0, "lol", "test.py", "assert_tensor_i8", 0 : tensor<32xi8>
  return
}

// -----
// CHECK-LABEL: assert_tensor_i64
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xi64>
func.func @assert_tensor_i64(%arg0: tensor<32xi64>) {
  // CHECK: %[[ASSERT:.*]] = linalg_ext.assert {msg = "test.py:0: assert_tensor_i64 Assertion `lol` failed"} ins(%[[ARG0]] : tensor<32xi64>) -> tensor<32xi64>
  tt.assert %arg0, "lol", "test.py", "assert_tensor_i64", 0 : tensor<32xi64>
  return
}

// -----
// CHECK-LABEL: @print_scalar
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: !tt.ptr<f32>, %[[ARG3:.*]]: i64)
// CHECK: %[[ARG4:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : !tt.ptr<f32> to i64
// CHECK: %[[ARG5:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG6:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG7:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG5]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG6]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG7]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print {format = "print test\0A"}
// CHECK: %[[ARG8:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG9:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG10:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG8]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG9]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG10]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = ""}
// CHECK: %[[ARG11:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG12:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG13:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG11]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG12]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG13]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = "arg0: \0A"}
// CHECK: %[[ARG14:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG15:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG16:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG14]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG15]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG16]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = "arg0, arg1"}
// CHECK: aux.scalar.print(%[[ARG1]] : f32) {format = "arg0, arg1"}
// CHECK: %[[ARG17:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG18:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG19:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG17]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG18]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG19]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG4]] : i64) {format = "arg2: %p"}
// CHECK: %[[ARG20:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG21:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG22:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG20]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG21]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG22]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG3]] : i64) {format = "arg3: \0A"}
func.func @print_scalar(%arg0: i32, %arg1: f32, %arg2: !tt.ptr<f32>, %arg3: i64) {
  tt.print "print test\n" { hex = false }
  tt.print "" { hex = false } : %arg0 : i32
  tt.print "arg0: \n" { hex = false } : %arg0 : i32
  tt.print "arg0, arg1" { hex = false } : %arg0, %arg1 : i32, f32
  tt.print "arg2: " { hex = false } : %arg2 : !tt.ptr<f32>
  tt.print "arg3: \n" { hex = false } : %arg3 : i64
  return
}

// -----
// CHECK-LABEL: @print_tensor
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xi32>, %[[ARG1:.*]]: tensor<2x8xf32>, %[[ARG2:.*]]: tensor<16x!tt.ptr<f32>>, %[[ARG3:.*]]: tensor<32xi64>)
// CHECK: %[[ARG4:.*]] = builtin.unrealized_conversion_cast %arg2 : tensor<16x!tt.ptr<f32>> to tensor<16xi64>
// CHECK: %[[ARG5:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG6:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG7:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG5]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG6]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG7]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG8:.*]] = aux.print(%[[ARG0]] : tensor<16xi32>) {format = ""} -> (tensor<16xi32>)
// CHECK: %[[ARG9:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG10:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG11:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG9]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG10]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG11]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG12:.*]] = aux.print(%[[ARG8]] : tensor<16xi32>) {format = "arg0: "} -> (tensor<16xi32>)
// CHECK: %[[ARG13:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG14:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG15:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG13]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG14]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG15]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG16:.*]] = aux.print(%[[ARG12]] : tensor<16xi32>) {format = "arg0, arg1"} -> (tensor<16xi32>)
// CHECK: %[[ARG17:.*]] = aux.print(%[[ARG1]] : tensor<2x8xf32>) {format = "arg0, arg1"} -> (tensor<2x8xf32>)
// CHECK: %[[ARG18:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG19:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG20:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG18]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG19]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG20]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG21:.*]] = aux.print(%[[ARG4]] : tensor<16xi64>) {format = "arg2: %p"} -> (tensor<16xi64>)
// CHECK: %[[ARG22:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG23:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG24:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG22]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG23]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG24]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG25:.*]] = aux.print(%[[ARG3]] : tensor<32xi64>) {format = ""} -> (tensor<32xi64>)
func.func @print_tensor(%arg0: tensor<16xi32>, %arg1: tensor<2x8xf32>, %arg2: tensor<16x!tt.ptr<f32>>, %arg3: tensor<32xi64>) {
  tt.print "" { hex = false } : %arg0 : tensor<16xi32>
  tt.print "arg0: " { hex = false } : %arg0 : tensor<16xi32>
  tt.print "arg0, arg1" { hex = false } : %arg0, %arg1 : tensor<16xi32>, tensor<2x8xf32>
  tt.print "arg2: " { hex = false } : %arg2 : tensor<16x!tt.ptr<f32>>
  tt.print "" { hex = false } : %arg3 : tensor<32xi64>
  return
}

// -----
// CHECK-LABEL: @print_scalar_and_tensor
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: tensor<16xi32>, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: tensor<2x8xf32>, %[[ARG5:.*]]: tensor<16x!tt.ptr<f32>>, %[[ARG6:.*]]: !tt.ptr<f32>, %[[ARG7:.*]]: i64, %[[ARG8:.*]]: tensor<32xi64>)
// CHECK: %[[ARG9:.*]] = builtin.unrealized_conversion_cast %arg5 : tensor<16x!tt.ptr<f32>> to tensor<16xi64>
// CHECK: %[[ARG10:.*]] = builtin.unrealized_conversion_cast %arg6 : !tt.ptr<f32> to i64
// CHECK: %[[ARG11:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG12:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG13:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG11]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG12]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG13]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = ""}
// CHECK: %[[ARG14:.*]] = aux.print(%[[ARG1]] : tensor<16xi32>) {format = ""} -> (tensor<16xi32>)
// CHECK: %[[ARG15:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG16:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG17:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG15]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG16]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG17]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = "arg0, arg1, arg2"}
// CHECK: %[[ARG18:.*]] = aux.print(%[[ARG14]] : tensor<16xi32>) {format = "arg0, arg1, arg2"} -> (tensor<16xi32>)
// CHECK: aux.scalar.print(%[[ARG2]] : f32) {format = "arg0, arg1, arg2"}
// CHECK: %[[ARG19:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG20:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG21:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG19]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG20]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG21]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = "arg0, arg1, arg2"}
// CHECK: %[[ARG22:.*]] = aux.print(%[[ARG18]] : tensor<16xi32>) {format = "arg0, arg1, arg2"} -> (tensor<16xi32>)
// CHECK: aux.scalar.print(%[[ARG2]] : f32) {format = "arg0, arg1, arg2"}
// CHECK: %[[ARG23:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG24:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG25:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG23]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG24]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG25]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = ""}
// CHECK: aux.scalar.print(%[[ARG2]] : f32) {format = ""}
// CHECK: %[[ARG26:.*]] = aux.print(%[[ARG22]] : tensor<16xi32>) {format = ""} -> (tensor<16xi32>)
// CHECK: %[[ARG27:.*]] = aux.print(%[[ARG4]] : tensor<2x8xf32>) {format = ""} -> (tensor<2x8xf32>)
// CHECK: aux.scalar.print(%[[ARG3]] : i32) {format = ""}
// CHECK: %[[ARG28:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG29:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG30:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG28]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG29]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG30]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG31:.*]] = aux.print(%[[ARG26]] : tensor<16xi32>) {format = "arg1, arg2, arg4, arg3"} -> (tensor<16xi32>)
// CHECK: aux.scalar.print(%[[ARG2]] : f32) {format = "arg1, arg2, arg4, arg3"}
// CHECK: %[[ARG32:.*]] = aux.print(%[[ARG27]] : tensor<2x8xf32>) {format = "arg1, arg2, arg4, arg3"} -> (tensor<2x8xf32>)
// CHECK: aux.scalar.print(%[[ARG3]] : i32) {format = "arg1, arg2, arg4, arg3"}
// CHECK: %[[ARG33:.*]] = tt.get_program_id x : i32
// CHECK: %[[ARG34:.*]] = tt.get_program_id y : i32
// CHECK: %[[ARG35:.*]] = tt.get_program_id z : i32
// CHECK: aux.scalar.print(%[[ARG33]] : i32) {format = "pid ("}
// CHECK: aux.scalar.print(%[[ARG34]] : i32) {format = ", "}
// CHECK: aux.scalar.print(%[[ARG35]] : i32) {format = ", "}
// CHECK: aux.scalar.print {format = ") "}
// CHECK: %[[ARG36:.*]] = aux.print(%[[ARG31]] : tensor<16xi32>) {format = "arg1, arg5, arg0, arg6, arg7, arg8"} -> (tensor<16xi32>)
// CHECK: %[[ARG37:.*]] = aux.print(%[[ARG9]] : tensor<16xi64>) {format = "arg1, arg5, arg0, arg6, arg7, arg8%p"} -> (tensor<16xi64>)
// CHECK: aux.scalar.print(%[[ARG0]] : i32) {format = "arg1, arg5, arg0, arg6, arg7, arg8%p"}
// CHECK: aux.scalar.print(%[[ARG10]] : i64) {format = "arg1, arg5, arg0, arg6, arg7, arg8%p"}
// CHECK: aux.scalar.print(%[[ARG7]] : i64) {format = "arg1, arg5, arg0, arg6, arg7, arg8%p"}
// CHECK: %[[ARG38:.*]] = aux.print(%[[ARG8]] : tensor<32xi64>) {format = "arg1, arg5, arg0, arg6, arg7, arg8%p"} -> (tensor<32xi64>)
func.func @print_scalar_and_tensor(%arg0: i32, %arg1: tensor<16xi32>, %arg2: f32, %arg3: i32, %arg4: tensor<2x8xf32>, %arg5: tensor<16x!tt.ptr<f32>>, %arg6: !tt.ptr<f32>, %arg7: i64, %arg8: tensor<32xi64>) {
  tt.print "" { hex = false } : %arg0, %arg1 : i32, tensor<16xi32>
  tt.print "arg0, arg1, arg2" { hex = false } : %arg0, %arg1, %arg2 : i32, tensor<16xi32>, f32
  tt.print "arg0, arg1, arg2" { hex = false } : %arg0, %arg1, %arg2 : i32, tensor<16xi32>, f32
  tt.print "" { hex = false } : %arg0, %arg2, %arg1, %arg4, %arg3 : i32, f32, tensor<16xi32>, tensor<2x8xf32>, i32
  tt.print "arg1, arg2, arg4, arg3" { hex = false } : %arg1, %arg2, %arg4, %arg3 : tensor<16xi32>, f32, tensor<2x8xf32>, i32
  tt.print "arg1, arg5, arg0, arg6, arg7, arg8" { hex = false } : %arg1, %arg5, %arg0, %arg6, %arg7, %arg8 : tensor<16xi32>, tensor<16x!tt.ptr<f32>>, i32, !tt.ptr<f32>, i64, tensor<32xi64>
  return
}

// -----
// CHECK-LABEL: @scan_add_2d_i32(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<1x2048xi32>) -> tensor<1x2048xi32> {
tt.func @scan_add_2d_i32(%arg0: tensor<1x2048xi32>) -> tensor<1x2048xi32> {
  // CHECK: %[[SCAN_INPUT:.*]] = tensor.extract_slice %[[INPUT]][0, 1] [1, 2047] [1, 1] : tensor<1x2048xi32> to tensor<1x2047xi32>
  // CHECK-NEXT: %[[SCAN_OUTPUT:.*]] = tensor.empty() : tensor<1x2047xi32>
  // CHECK-NEXT: %[[INIT:.*]] = tensor.extract_slice %[[INPUT]][0, 0] [1, 1] [1, 1] : tensor<1x2048xi32> to tensor<1x1xi32>
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[SHAPE:.*]] = tensor.from_elements %[[C1]] : tensor<1xindex>
  // CHECK-NEXT: %[[SCAN_INIT:.*]] = tensor.reshape %[[INIT]](%[[SHAPE]]) : (tensor<1x1xi32>, tensor<1xindex>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = linalg_ext.scan ins(%[[SCAN_INPUT]] : tensor<1x2047xi32>) outs(%[[SCAN_OUTPUT]], %[[SCAN_INIT]] : tensor<1x2047xi32>, tensor<1xi32>) dimensions = [1] {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: i32, %[[OUTPUT:.*]]: i32, %[[INIT:.*]]: i32):
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[IN]], %[[INIT]] : i32
  // CHECK-NEXT: linalg_ext.yield %[[ADD]], %[[ADD]] : i32, i32
  // CHECK-NEXT: } -> tensor<1x2047xi32>, tensor<1xi32>
  // CHECK-NEXT: %[[INERT_SLICE:.*]] = tensor.insert_slice %[[SCAN]]#0 into %[[INPUT]][0, 1] [1, 2047] [1, 1] : tensor<1x2047xi32> into tensor<1x2048xi32>
  // CHECK-NEXT: return %[[INERT_SLICE]] : tensor<1x2048xi32>

  %0 = "tt.scan" (%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) {axis = 1 : i32, reverse = false} : (tensor<1x2048xi32>) -> tensor<1x2048xi32>
  tt.return  %0 : tensor<1x2048xi32>
}

// -----
// CHECK-LABEL: @scan_min_2d_f16(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<1x2048xf16>) -> tensor<1x2048xf16> {
tt.func @scan_min_2d_f16(%arg0: tensor<1x2048xf16>) -> tensor<1x2048xf16> {
  // CHECK: %[[SCAN_INPUT:.*]] = tensor.extract_slice %[[INPUT]][0, 1] [1, 2047] [1, 1] : tensor<1x2048xf16> to tensor<1x2047xf16>
  // CHECK-NEXT: %[[SCAN_OUTPUT:.*]] = tensor.empty() : tensor<1x2047xf16>
  // CHECK-NEXT: %[[INIT:.*]] = tensor.extract_slice %[[INPUT]][0, 0] [1, 1] [1, 1] : tensor<1x2048xf16> to tensor<1x1xf16>
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[SHAPE:.*]] = tensor.from_elements %[[C1]] : tensor<1xindex>
  // CHECK-NEXT: %[[SCAN_INIT:.*]] = tensor.reshape %[[INIT]](%[[SHAPE]]) : (tensor<1x1xf16>, tensor<1xindex>) -> tensor<1xf16>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = linalg_ext.scan ins(%[[SCAN_INPUT]] : tensor<1x2047xf16>) outs(%[[SCAN_OUTPUT]], %[[SCAN_INIT]] : tensor<1x2047xf16>, tensor<1xf16>) dimensions = [1] {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: f16, %[[OUTPUT:.*]]: f16, %[[INIT:.*]]: f16):
  // CHECK-NEXT: %[[MIN:.*]] = arith.minimumf %[[IN]], %[[INIT]] : f16
  // CHECK-NEXT: linalg_ext.yield %[[MIN]], %[[MIN]] : f16, f16
  // CHECK-NEXT: } -> tensor<1x2047xf16>, tensor<1xf16>
  // CHECK-NEXT: %[[INERT_SLICE:.*]] = tensor.insert_slice %[[SCAN]]#0 into %[[INPUT]][0, 1] [1, 2047] [1, 1] : tensor<1x2047xf16> into tensor<1x2048xf16>
  // CHECK-NEXT: return %[[INERT_SLICE]] : tensor<1x2048xf16>

  %0 = "tt.scan" (%arg0) ({
  ^bb0(%arg1: f16, %arg2: f16):
    %1 = arith.minimumf %arg1, %arg2 : f16
    tt.scan.return %1 : f16
  }) {axis = 1 : i32, reverse = false} : (tensor<1x2048xf16>) -> tensor<1x2048xf16>
  tt.return %0 : tensor<1x2048xf16>
}

// -----
// CHECK-LABEL: @scan_add_1d_size1_f32(
// CHECK-SAME:                         %[[INPUT:.*]]: tensor<1xf32>
tt.func @scan_add_1d_size1_f32(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: return %[[INPUT]] : tensor<1xf32>
  %0 = "tt.scan" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.scan.return %1 : f32
  }) {axis = 0 : i32, reverse = false} : (tensor<1xf32>) -> tensor<1xf32>
  tt.return %0 : tensor<1xf32>
}

// -----
// CHECK-LABEL: @scan_sub_1d_f32(
// CHECK-SAME:                   %[[INPUT:.*]]: tensor<2048xf32>) -> tensor<2048xf32> {
tt.func @scan_sub_1d_f32(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
  // CHECK: %[[SCAN_INPUT:.*]] = tensor.extract_slice %[[INPUT]][1] [2047] [1] : tensor<2048xf32> to tensor<2047xf32>
  // CHECK-NEXT: %[[SCAN_OUTPUT:.*]] = tensor.empty() : tensor<2047xf32>
  // CHECK-NEXT: %[[INIT:.*]] = tensor.extract_slice %[[INPUT]][0] [1] [1] : tensor<2048xf32> to tensor<1xf32>
  // CHECK-NEXT: %[[SHAPE:.*]] = tensor.from_elements : tensor<0xindex>
  // CHECK-NEXT: %[[SCAN_INIT:.*]] = tensor.reshape %[[INIT]](%[[SHAPE]]) : (tensor<1xf32>, tensor<0xindex>) -> tensor<f32>
  // CHECK-NEXT: %[[SCAN:.*]]:2 = linalg_ext.scan ins(%[[SCAN_INPUT]] : tensor<2047xf32>) outs(%[[SCAN_OUTPUT]], %[[SCAN_INIT]] : tensor<2047xf32>, tensor<f32>) dimensions = [0] {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[OUTPUT:.*]]: f32, %[[INIT:.*]]: f32):
  // CHECK-NEXT: %[[SUB:.*]] = arith.subf %[[IN]], %[[INIT]] : f32
  // CHECK-NEXT: linalg_ext.yield %[[SUB]], %[[SUB]] : f32, f32
  // CHECK-NEXT: } -> tensor<2047xf32>, tensor<f32>
  // CHECK-NEXT: %[[INERT_SLICE:.*]] = tensor.insert_slice %[[SCAN]]#0 into %[[INPUT]][1] [2047] [1] : tensor<2047xf32> into tensor<2048xf32>
  // CHECK-NEXT: return %[[INERT_SLICE]] : tensor<2048xf32>

  %0 = "tt.scan" (%arg0) ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.subf %arg1, %arg2 : f32
    tt.scan.return %1 : f32
  }) {axis = 0 : i32, reverse = false} : (tensor<2048xf32>) -> tensor<2048xf32>
  tt.return %0 : tensor<2048xf32>
}
