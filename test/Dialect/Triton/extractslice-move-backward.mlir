// RUN: triton-linalg-opt --extract-like-move-backward --canonicalize -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @extract_slice_from_fill(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                       %[[VAL_1:.*]]: index) -> tensor<?xi64> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty(%[[VAL_1]]) : tensor<?x1xi64>
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1]] : tensor<?x1xi64> into tensor<?xi64>
// CHECK:           %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_0]] : i64) outs(%[[VAL_3]] : tensor<?xi64>) -> tensor<?xi64>
// CHECK:           return %[[VAL_4]] : tensor<?xi64>
// CHECK:         }
func.func @extract_slice_from_fill(%arg0: i64, %arg1: index) -> tensor<?xi64> {
  %0 = tensor.empty() : tensor<128x16xi64>
  %1 = linalg.fill ins(%arg0 : i64) outs(%0 : tensor<128x16xi64>) -> tensor<128x16xi64>
  %2 = tensor.extract_slice %1[8, 0] [%arg1, 1] [4, 1] : tensor<128x16xi64> to tensor<?xi64>
  return %2 : tensor<?xi64>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_broadcast_op(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<128x4xi32>) -> tensor<100x4xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][10, 0] [100, 1] [1, 1] : tensor<128x4xi32> to tensor<100x1xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<100x4x1xi32>
// CHECK:           %[[VAL_3:.*]] = linalg.broadcast ins(%[[VAL_1]] : tensor<100x1xi32>) outs(%[[VAL_2]] : tensor<100x4x1xi32>) dimensions = [1]
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0], [1, 2]] : tensor<100x4x1xi32> into tensor<100x4xi32>
// CHECK:           return %[[VAL_4]] : tensor<100x4xi32>
// CHECK:         }
func.func @extract_slice_from_broadcast_op(%arg0: tensor<128x4xi32>) -> tensor<100x4xi32> {
  %1 = tensor.empty() : tensor<128x16x4xi32>
  %2 = linalg.broadcast ins(%arg0: tensor<128x4xi32>) outs(%1: tensor<128x16x4xi32>) dimensions = [1]
  %3 = tensor.extract_slice %2[10, 5, 0] [100, 4, 1] [1, 2, 1] : tensor<128x16x4xi32> to tensor<100x4xi32>
  return %3 : tensor<100x4xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_expand_shape_op1(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<5x6x7xi32>) -> tensor<3x2xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 2] [3, 1, 2] [1, 1, 1] : tensor<5x6x7xi32> to tensor<3x1x2xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1], [2]] : tensor<3x1x2xi32> into tensor<3x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<3x2xi32>
// CHECK:         }
func.func @extract_slice_from_expand_shape_op1(%arg0: tensor<5x6x7xi32>) -> tensor<3x2xi32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<5x6x7xi32> into tensor<5x6x1x7xi32>
  %1 = tensor.extract_slice %0[1, 0, 0, 2] [3, 1, 1, 2] [1, 1, 1, 1] : tensor<5x6x1x7xi32> to tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_expand_shape_op2(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<5x6x7xi32>) -> tensor<3x2x3xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 2] [3, 6, 1] [1, 1, 1] : tensor<5x6x7xi32> to tensor<3x6x1xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0], [1, 2], [3]] : tensor<3x6x1xi32> into tensor<3x2x3x1xi32>
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0], [1], [2, 3]] : tensor<3x2x3x1xi32> into tensor<3x2x3xi32>
// CHECK:           return %[[VAL_3]] : tensor<3x2x3xi32>
// CHECK:         }
func.func @extract_slice_from_expand_shape_op2(%arg0: tensor<5x6x7xi32>) -> tensor<3x2x3xi32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<5x6x7xi32> into tensor<5x2x3x7xi32>
  %1 = tensor.extract_slice %0[1, 0, 0, 2] [3, 2, 3, 1] [1, 1, 1, 1] : tensor<5x2x3x7xi32> to tensor<3x2x3xi32>
  return %1 : tensor<3x2x3xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_expand_shape_op3(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<5x6x7xi32>) -> tensor<3x2x2xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1, 2], [3]] : tensor<5x6x7xi32> into tensor<5x2x3x7xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_1]][1, 0, 0, 2] [3, 1, 2, 2] [1, 1, 1, 1] : tensor<5x2x3x7xi32> to tensor<3x2x2xi32>
// CHECK:           return %[[VAL_2]] : tensor<3x2x2xi32>
// CHECK:         }
func.func @extract_slice_from_expand_shape_op3(%arg0: tensor<5x6x7xi32>) -> tensor<3x2x2xi32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] : tensor<5x6x7xi32> into tensor<5x2x3x7xi32>
  %1 = tensor.extract_slice %0[1, 0, 0, 2] [3, 1, 2, 2] [1, 1, 1, 1] : tensor<5x2x3x7xi32> to tensor<3x2x2xi32>
  return %1 : tensor<3x2x2xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_collapse_shape_op1(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<5x2x3x7xi32>) -> tensor<6x7xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [1, 2, 3, 7] [1, 1, 1, 1] : tensor<5x2x3x7xi32> to tensor<1x2x3x7xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1, 2], [3]] : tensor<1x2x3x7xi32> into tensor<6x7xi32>
// CHECK:           return %[[VAL_2]] : tensor<6x7xi32>
// CHECK:         }
func.func @extract_slice_from_collapse_shape_op1(%arg0: tensor<5x2x3x7xi32>) -> tensor<6x7xi32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<5x2x3x7xi32> into tensor<5x6x7xi32>
  %1 = tensor.extract_slice %0[0, 0, 0] [1, 6, 7] [1, 1, 1] : tensor<5x6x7xi32> to tensor<6x7xi32>
  return %1 : tensor<6x7xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_collapse_shape_op2(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<5x2x3x7xi32>) -> tensor<6xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [1, 2, 3, 1] [1, 1, 1, 1] : tensor<5x2x3x7xi32> to tensor<1x2x3x1xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1, 2, 3]] : tensor<1x2x3x1xi32> into tensor<6xi32>
// CHECK:           return %[[VAL_2]] : tensor<6xi32>
// CHECK:         }
func.func @extract_slice_from_collapse_shape_op2(%arg0: tensor<5x2x3x7xi32>) -> tensor<6xi32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<5x2x3x7xi32> into tensor<5x6x7xi32>
  %1 = tensor.extract_slice %0[0, 0, 0] [1, 6, 1] [1, 1, 1] : tensor<5x6x7xi32> to tensor<6xi32>
  return %1 : tensor<6xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_collapse_shape_op3(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<5x2x3x7xi32>) -> tensor<2x7xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2], [3]] : tensor<5x2x3x7xi32> into tensor<5x6x7xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_1]][0, 0, 0] [1, 2, 7] [1, 1, 1] : tensor<5x6x7xi32> to tensor<2x7xi32>
// CHECK:           return %[[VAL_2]] : tensor<2x7xi32>
// CHECK:         }
func.func @extract_slice_from_collapse_shape_op3(%arg0: tensor<5x2x3x7xi32>) -> tensor<2x7xi32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<5x2x3x7xi32> into tensor<5x6x7xi32>
  %1 = tensor.extract_slice %0[0, 0, 0] [1, 2, 7] [1, 1, 1] : tensor<5x6x7xi32> to tensor<2x7xi32>
  return %1 : tensor<2x7xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_collapse_shape_op4(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<?x2x3xi32>) -> tensor<?xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x2x3xi32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] {{\[}}%[[VAL_2]], 2, 1] [1, 1, 1] : tensor<?x2x3xi32> to tensor<?x2x1xi32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0, 1, 2]] : tensor<?x2x1xi32> into tensor<?xi32>
// CHECK:           return %[[VAL_4]] : tensor<?xi32>
// CHECK:         }
func.func @extract_slice_from_collapse_shape_op4(%arg0: tensor<?x2x3xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x2x3xi32> into tensor<?x3xi32>
  %1 = tensor.dim %0, %c0 : tensor<?x3xi32>
  %2 = tensor.extract_slice %0[0, 0] [%1, 1] [1, 1] : tensor<?x3xi32> to tensor<?xi32>
  return %2 : tensor<?xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_collapse_shape_op_with_0_rank(
// CHECK-SAME:                                                                %[[VAL_0:.*]]: tensor<1x1xf16>) -> tensor<f16> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] [] : tensor<1x1xf16> into tensor<f16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.broadcast ins(%[[VAL_1]] : tensor<f16>) outs(%[[VAL_2]] : tensor<1xf16>) dimensions = [0]
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_3]] [] : tensor<1xf16> into tensor<f16>
// CHECK:           return %[[VAL_4]] : tensor<f16>
// CHECK:         }
func.func @extract_slice_from_collapse_shape_op_with_0_rank(%arg0 : tensor<1x1xf16>) -> tensor<f16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x1xf16> into tensor<1xf16>
  %collapsed_1 = tensor.collapse_shape %collapsed [] : tensor<1xf16> into tensor<f16>
  %0 = tensor.empty() : tensor<256xf16>
  %broadcasted = linalg.broadcast ins(%collapsed_1 : tensor<f16>) outs(%0 : tensor<256xf16>) dimensions = [0]
  %extracted_slice = tensor.extract_slice %broadcasted[0] [1] [1] : tensor<256xf16> to tensor<f16>
  return %extracted_slice : tensor<f16>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_map_arith_add(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<128x16xi32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<128x16xi32>) -> tensor<16xi32> {
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [16, 1] [2, 2] : tensor<128x16xi32> to tensor<16x1xi32>
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_1]][0, 0] [16, 1] [2, 2] : tensor<128x16xi32> to tensor<16x1xi32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<16x1xi32>
// CHECK:           %[[VAL_5:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[VAL_2]], %[[VAL_3]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_4]] : tensor<16x1xi32>)
// CHECK:           %[[VAL_6:.*]] = tensor.collapse_shape %[[VAL_5]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           return %[[VAL_6]] : tensor<16xi32>
// CHECK:         }
func.func @extract_slice_from_map_arith_add(
    %arg0: tensor<128x16xi32>, %arg1: tensor<128x16xi32>) -> tensor<16xi32> {
  %0 = tensor.empty() : tensor<128x16xi32>
  %1 = linalg.map { arith.addi } ins(%arg0, %arg1 : tensor<128x16xi32>, tensor<128x16xi32>) outs(%0 : tensor<128x16xi32>)
  %2 = tensor.extract_slice %1[0, 0] [16, 1] [2, 2] : tensor<128x16xi32> to tensor<16xi32>
  return %2 : tensor<16xi32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_select_op(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<128x16xi1>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<128x16xf32>,
// CHECK-SAME:                                            %[[VAL_2:.*]]: tensor<128x16xf32>) -> tensor<100xf32> {
// CHECK:           %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][10, 5] [100, 1] [1, 2] : tensor<128x16xi1> to tensor<100x1xi1>
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_1]][10, 5] [100, 1] [1, 2] : tensor<128x16xf32> to tensor<100x1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_2]][10, 5] [100, 1] [1, 2] : tensor<128x16xf32> to tensor<100x1xf32>
// CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : tensor<100x1xi1>, tensor<100x1xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.collapse_shape %[[VAL_6]] {{\[\[}}0, 1]] : tensor<100x1xf32> into tensor<100xf32>
// CHECK:           return %[[VAL_7]] : tensor<100xf32>
// CHECK:         }
func.func @extract_slice_from_select_op(%arg0: tensor<128x16xi1>, %arg1: tensor<128x16xf32>, %arg2: tensor<128x16xf32>) -> tensor<100xf32> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<128x16xi1>, tensor<128x16xf32>
  %1 = tensor.extract_slice %0[10, 5] [100, 1] [1, 2] : tensor<128x16xf32> to tensor<100xf32>
  return %1 : tensor<100xf32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_constant() -> tensor<i32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant dense<3> : tensor<i32>
// CHECK:           return %[[VAL_0]] : tensor<i32>
// CHECK:         }
func.func @extract_slice_from_constant() -> tensor<i32> {
  %cst = arith.constant dense<3> : tensor<6xi32>
  %extracted = tensor.extract_slice %cst[0] [1] [1] : tensor<6xi32> to tensor<i32>
  return %extracted : tensor<i32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_for_iter_args(
// CHECK-SAME:                                                %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<64x64xf32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_7]] : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:           %[[VAL_9:.*]]:2 = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_11:.*]] = %[[VAL_1]], %[[VAL_12:.*]] = %[[VAL_2]]) -> (tensor<64x64xf32>, tensor<32x32xf32>) {
// CHECK:             %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_11]][0, 0] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
// CHECK:             %[[VAL_14:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK:             %[[VAL_15:.*]] = linalg.map { arith.addf } ins(%[[VAL_13]], %[[VAL_12]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_14]] : tensor<32x32xf32>)
// CHECK:             %[[VAL_16:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:             %[[VAL_17:.*]] = linalg.map { arith.addf } ins(%[[VAL_11]], %[[VAL_8]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[VAL_16]] : tensor<64x64xf32>)
// CHECK:             scf.yield %[[VAL_17]], %[[VAL_15]] : tensor<64x64xf32>, tensor<32x32xf32>
// CHECK:           }
// CHECK:           return %[[VAL_18:.*]]#1 : tensor<32x32xf32>
// CHECK:         }
func.func @extract_slice_from_for_iter_args(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %2:2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<64x64xf32>, tensor<32x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg4[0, 0] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
    %3 = tensor.empty() : tensor<32x32xf32>
    %mapped = linalg.map { arith.addf } ins(%extracted_slice, %arg5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>)
    %4 = tensor.empty() : tensor<64x64xf32>
    %mapped_0 = linalg.map { arith.addf } ins(%arg4, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<64x64xf32>)
    scf.yield %mapped_0, %mapped : tensor<64x64xf32>, tensor<32x32xf32>
  }
  return %2#1 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL:   func.func @extract_slice_from_broadcast_op_with_0d_input(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<i32>) -> tensor<100xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<100x1xi32>
// CHECK:           %[[VAL_2:.*]] = linalg.broadcast ins(%[[VAL_0]] : tensor<i32>) outs(%[[VAL_1]] : tensor<100x1xi32>) dimensions = [0, 1]
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1]] : tensor<100x1xi32> into tensor<100xi32>
// CHECK:           return %[[VAL_3]] : tensor<100xi32>
// CHECK:         }
func.func @extract_slice_from_broadcast_op_with_0d_input(%arg0: tensor<i32>) -> tensor<100xi32> {
  %0 = tensor.empty() : tensor<128x16xi32>
  %1 = linalg.broadcast ins(%arg0: tensor<i32>) outs(%0: tensor<128x16xi32>) dimensions = [0, 1]
  %2 = tensor.extract_slice %1[10, 5] [100, 1] [1, 2] : tensor<128x16xi32> to tensor<100xi32>
  return %2 : tensor<100xi32>
}

// -----
// CHECK-LABEL:   func.func @test_destination_style_op_for_result_chain(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_1]] : tensor<64xf32>)
// CHECK:           %[[VAL_3:.*]] = linalg.map { math.atan } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_2]] : tensor<64xf32>)
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.absf } ins(%[[VAL_4]] : tensor<1xf32>) outs(%[[VAL_5]] : tensor<1xf32>)
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.map { math.exp } ins(%[[VAL_6]] : tensor<1xf32>) outs(%[[VAL_7]] : tensor<1xf32>)
// CHECK:           %[[VAL_9:.*]] = tensor.collapse_shape %[[VAL_8]] [] : tensor<1xf32> into tensor<f32>
// CHECK:           return %[[VAL_3]], %[[VAL_9]] : tensor<64xf32>, tensor<f32>
// CHECK:         }
func.func @test_destination_style_op_for_result_chain(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %4 = tensor.extract_slice %3[0] [1] [1] : tensor<64xf32> to tensor<f32>
  return %2, %4 : tensor<64xf32>, tensor<f32>
}

// -----
// CHECK-LABEL:   func.func @test_destination_style_op_for_init_chain(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_1]] : tensor<64xf32>)
// CHECK:           %[[VAL_3:.*]] = linalg.map { math.exp } ins(%[[VAL_2]] : tensor<64xf32>) outs(%[[VAL_2]] : tensor<64xf32>)
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.atan } ins(%[[VAL_4]] : tensor<1xf32>) outs(%[[VAL_5]] : tensor<1xf32>)
// CHECK:           %[[VAL_7:.*]] = tensor.collapse_shape %[[VAL_6]] [] : tensor<1xf32> into tensor<f32>
// CHECK:           return %[[VAL_3]], %[[VAL_7]] : tensor<64xf32>, tensor<f32>
// CHECK:         }
func.func @test_destination_style_op_for_init_chain(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %4 = tensor.extract_slice %2[0] [1] [1] : tensor<64xf32> to tensor<f32>
  return %3, %4 : tensor<64xf32>, tensor<f32>
}

// -----
// CHECK-LABEL:   func.func @test_destination_style_op_cross_block(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_1]] : tensor<64xf32>)
// CHECK:           %[[VAL_3:.*]] = linalg.map { math.atan } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_2]] : tensor<64xf32>)
// CHECK:           %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0] [1] [1] : tensor<64xf32> to tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.absf } ins(%[[VAL_4]] : tensor<1xf32>) outs(%[[VAL_5]] : tensor<1xf32>)
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.map { math.exp } ins(%[[VAL_6]] : tensor<1xf32>) outs(%[[VAL_7]] : tensor<1xf32>)
// CHECK:           %[[VAL_9:.*]] = tensor.collapse_shape %[[VAL_8]] [] : tensor<1xf32> into tensor<f32>
// CHECK:           return %[[VAL_3]], %[[VAL_9]] : tensor<64xf32>, tensor<f32>
// CHECK:         }
func.func @test_destination_style_op_cross_block(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<f32>) {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  cf.br ^bb1
^bb1:
  %4 = tensor.extract_slice %3[0] [1] [1] : tensor<64xf32> to tensor<f32>
  return %2, %4 : tensor<64xf32>, tensor<f32>
}

// -----
// CHECK-LABEL:   func.func @for_op_with_0d_iter_args(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_7]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_9:.*]]:2 = scf.for %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_11:.*]] = %[[VAL_1]], %[[VAL_12:.*]] = %[[VAL_2]]) -> (tensor<f32>, tensor<f32>) {
// CHECK:             %[[VAL_13:.*]] = tensor.empty() : tensor<f32>
// CHECK:             %[[VAL_14:.*]] = linalg.map { arith.addf } ins(%[[VAL_11]], %[[VAL_12]] : tensor<f32>, tensor<f32>) outs(%[[VAL_13]] : tensor<f32>)
// CHECK:             %[[VAL_15:.*]] = tensor.empty() : tensor<f32>
// CHECK:             %[[VAL_16:.*]] = linalg.map { arith.addf } ins(%[[VAL_11]], %[[VAL_8]] : tensor<f32>, tensor<f32>) outs(%[[VAL_15]] : tensor<f32>)
// CHECK:             scf.yield %[[VAL_16]], %[[VAL_14]] : tensor<f32>, tensor<f32>
// CHECK:           }
// CHECK:           return %[[VAL_17:.*]]#1 : tensor<f32>
// CHECK:         }
func.func @for_op_with_0d_iter_args(%arg0: i64, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  %2:2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<f32>, tensor<f32>) {
    %extracted_slice = tensor.extract_slice %arg4[] [] [] : tensor<f32> to tensor<f32>
    %3 = tensor.empty() : tensor<f32>
    %mapped = linalg.map { arith.addf } ins(%extracted_slice, %arg5 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>)
    %4 = tensor.empty() : tensor<f32>
    %mapped_0 = linalg.map { arith.addf } ins(%arg4, %1 : tensor<f32>, tensor<f32>) outs(%4 : tensor<f32>)
    scf.yield %mapped_0, %mapped : tensor<f32>, tensor<f32>
  }
  return %2#1 : tensor<f32>
}

// -----
// CHECK-LABEL: func.func @extractslice_outside_failed
// CHECK-NOT: tensor.extract_slice
// CHECK:     scf.for
func.func @extractslice_outside_failed(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
  %memref = aux.view %ptr to offset: [0], sizes: [32, 64], strides: [64, 1] : !llvm.ptr<1> to memref<32x64xf32, 1>
  %3 = bufferization.to_tensor %memref : memref<32x64xf32, 1>
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %1 = tensor.extract_slice %arg5[%c0, %c0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
    %2 = tensor.extract_slice %arg6[%c0, %c0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
    %4 = tensor.empty() : tensor<32x64xf32>
    %mapped = linalg.map { arith.addf } ins(%1, %3 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%4 : tensor<32x64xf32>)
    %5 = tensor.insert_slice %mapped into %arg5[%c0, %c0] [32, 64] [1, 1] : tensor<32x64xf32> into tensor<64x64xf32>
    %6 = tensor.empty() : tensor<64x64xf32>
    %transpose = linalg.transpose ins(%5 : tensor<64x64xf32>) outs(%6 : tensor<64x64xf32>) permutation = [1, 0]
    %7 = tensor.empty() : tensor<32x64xf32>
    %8 = linalg.map { arith.addf } ins(%2, %3 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%7 : tensor<32x64xf32>)
    %9 = tensor.insert_slice %8 into %arg6[%c0, %c0] [32, 64] [1, 1] : tensor<32x64xf32> into tensor<64x64xf32>
    scf.yield %transpose, %9 : tensor<64x64xf32>, tensor<64x64xf32>
  }
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----
// CHECK-LABEL:   func.func @extractslice_cross_iter_args(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<128x64xi32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<128x64xi32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_1]][0, 0] [128, 1] [1, 1] : tensor<128x64xi32> to tensor<128xi32>
// CHECK:           %[[VAL_6:.*]] = scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_8:.*]] = %[[VAL_5]]) -> (tensor<128xi32>)  : i32 {
// CHECK:             %[[VAL_9:.*]] = tensor.expand_shape %[[VAL_8]] {{\[\[}}0, 1]] : tensor<128xi32> into tensor<128x1xi32>
// CHECK:             "test.foo"(%[[VAL_8]]) : (tensor<128xi32>) -> ()
// CHECK:             %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [128, 1] [1, 1] : tensor<128x64xi32> to tensor<128x1xi32>
// CHECK:             %[[VAL_11:.*]] = tensor.empty() : tensor<128x1xi32>
// CHECK:             %[[VAL_12:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[VAL_9]], %[[VAL_10]] : tensor<128x1xi32>, tensor<128x1xi32>) outs(%[[VAL_11]] : tensor<128x1xi32>)
// CHECK:             %[[VAL_13:.*]] = tensor.collapse_shape %[[VAL_12]] {{\[\[}}0, 1]] : tensor<128x1xi32> into tensor<128xi32>
// CHECK:             scf.yield %[[VAL_13]] : tensor<128xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @extractslice_cross_iter_args(%arg0: tensor<128x64xi32>, %arg1: tensor<128x64xi32>) {
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<128x64xi32>)  : i32 {
    %extracted_slice = tensor.extract_slice %arg3[0, 0] [128, 1] [1, 1] : tensor<128x64xi32> to tensor<128xi32>
    "test.foo"(%extracted_slice) : (tensor<128xi32>) -> ()
    %1 = tensor.empty() : tensor<128x64xi32>
    %mapped = linalg.map { arith.addi } ins(%arg3, %arg0 : tensor<128x64xi32>, tensor<128x64xi32>) outs(%1 : tensor<128x64xi32>)
    scf.yield %mapped : tensor<128x64xi32>
  }
  return
}
