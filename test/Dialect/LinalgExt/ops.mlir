// RUN: triton-linalg-opt %s -split-input-file | FileCheck %s

func.func @transpose(%input: tensor<16x32x64xf32>,
                     %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}
// CHECK-LABEL: func @transpose

// -----

func.func @transpose_memref(%input: memref<16x32x64xf32>,
                            %init: memref<32x64x16xf32>) {
  linalg.transpose
      ins(%input:memref<16x32x64xf32>)
      outs(%init:memref<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return
}
// CHECK-LABEL: func @transpose_memref

// -----

func.func @map(%lhs: tensor<16x32x64xf32>, %rhs: tensor<16x32x64xf32>,
               %init: tensor<16x32x64xf32>) {
  linalg.map
      ins(%lhs, %rhs : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init: tensor<16x32x64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map

// -----

func.func @map_memref(%lhs: memref<16x32x64xf32>, %rhs: memref<16x32x64xf32>,
                      %init: memref<16x32x64xf32>) {
  linalg.map
      ins(%lhs, %rhs : memref<16x32x64xf32>, memref<16x32x64xf32>)
      outs(%init: memref<16x32x64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map_memref

// -----

func.func @reduce(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) {
  linalg.reduce
      ins(%input: tensor<16x32x64xf32>)
      outs(%init: tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @reduce

// -----

func.func @reduce_memref(%input: memref<16x32x64xf32>,
                         %init: memref<16x64xf32>) {
  linalg.reduce
      ins(%input: memref<16x32x64xf32>)
      outs(%init: memref<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @reduce_memref

// -----

func.func @broadcast(%input: tensor<16xf32>,
                     %init: tensor<16x64xf32>) {
  linalg.broadcast
      ins(%input: tensor<16xf32>)
      outs(%init: tensor<16x64xf32>)
      dimensions = [1]
  func.return
}
// CHECK-LABEL: func @broadcast

// -----

func.func @broadcast_memref(%input: memref<16xf32>,
                            %init: memref<16x64xf32>) {
  linalg.broadcast
      ins(%input: memref<16xf32>)
      outs(%init: memref<16x64xf32>)
      dimensions = [1]
  func.return
}
// CHECK-LABEL: func @broadcast_memref

// -----
// CHECK: linalg_ext.batch_conv_2d_nhwc_fhwc
func.func @batch_conv_2d_nhwc_fhwc_static(%input : tensor<8x128x1x1x256xf32>, %filter : tensor<8x256x1x1x256xf32>) -> (tensor<8x128x1x1x256xf32>) {
  %0 = tensor.empty () : tensor<8x128x1x1x256xf32>
  %1 = linalg_ext.batch_conv_2d_nhwc_fhwc
      ins(%input, %filter : tensor<8x128x1x1x256xf32>, tensor<8x256x1x1x256xf32>)
      outs(%0 : tensor<8x128x1x1x256xf32>) -> tensor<8x128x1x1x256xf32>
  return %1 : tensor<8x128x1x1x256xf32>
}

// -----
// CHECK: linalg_ext.batch_conv_2d_nhwc_fhwc
func.func @batch_conv_2d_nhwc_fhwc_dynamic(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg_ext.batch_conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----
// CHECK: linalg_ext.batch_conv_2d_nhwc_fhwc
func.func @batch_conv_2d_nhwc_fhwc_memref(%input: memref<2x32x1x1x256xf32, 101>, %filter: memref<2x64x1x1x256xf32, 101>, %init: memref<2x32x1x1x64xf32, 101>) -> memref<2x32x1x1x64xf32, 101> {
  "linalg_ext.batch_conv_2d_nhwc_fhwc"(%input, %filter, %init) ({
    ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
        "linalg.yield"(%arg6) : (f32) -> ()
      }) {__slimex_reorder_strategy__ = "forward64", linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 + d5, d3 + d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], operand_segment_sizes = array<i32: 2, 1>} : (memref<2x32x1x1x256xf32, 101>, memref<2x64x1x1x256xf32, 101>, memref<2x32x1x1x64xf32, 101>) -> ()
  return %init : memref<2x32x1x1x64xf32, 101>
}

// -----
// CHECK: linalg_ext.make_range
func.func @make_range_normal(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %0 = linalg_ext.make_range ins(%c0, %c128 : i32, i32) outs(%arg0:tensor<128xi32>) -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----
// CHECK: linalg_ext.make_range
func.func @make_range_start_not_zero(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %c2 = arith.constant 2 : i32
  %c130 = arith.constant 130 : i32
  %0 = linalg_ext.make_range ins(%c2, %c130 : i32, i32) outs(%arg0:tensor<128xi32>) -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----
// CHECK: linalg_ext.im2col
func.func @im2col(%input : tensor<128x16x16x256xf32>) -> (tensor<128x14x14x3x3x256xf32>) {
  %0 = tensor.empty () : tensor<128x14x14x3x3x256xf32>
  %1 = linalg_ext.im2col
      ins(%input : tensor<128x16x16x256xf32>)
      outs(%0 : tensor<128x14x14x3x3x256xf32>) -> tensor<128x14x14x3x3x256xf32>
  return %1 : tensor<128x14x14x3x3x256xf32>
}

// -----
// CHECK: linalg_ext.im2col
func.func @im2col_memref(%input: memref<32x16x16x256xf32, 101>, %init: memref<32x14x14x3x3x256xf32, 101>) -> memref<32x14x14x3x3x256xf32, 101> {
  "linalg_ext.im2col"(%input, %init) ({
    ^bb0(%arg6: f32, %arg7: f32):
        "linalg.yield"(%arg6) : (f32) -> ()
      }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d4, d5, d6) -> (d0, d1, d2, d4, d5, d6)>], operand_segment_sizes = array<i32: 1, 1>} : (memref<32x16x16x256xf32, 101>, memref<32x14x14x3x3x256xf32, 101>) -> ()
  return %init : memref<32x14x14x3x3x256xf32, 101>
}

// -----
// CHECK: linalg_ext.scatter
func.func @scatter_tensor(%A : tensor<4x1xi32>, %B: tensor<4x2x4xf32>, %C: tensor<16x8xf32>, %D: tensor<4xi1>) -> tensor<16x8xf32> {
  %scatter = linalg_ext.scatter
              dimension_map = [1]
              ranged_data(true)
              overlap_window(false) signed_indice(true)
              ins(%B, %A, %D: tensor<4x2x4xf32>, tensor<4x1xi32>, tensor<4xi1>)
              outs(%C: tensor<16x8xf32>) {
                ^bb0(%arg0 :f32, %arg1: f32):
                  linalg_ext.yield %arg0 : f32
              } -> tensor<16x8xf32>
  return %scatter : tensor<16x8xf32>
}

// -----
// CHECK: linalg_ext.scatter
func.func @scatter_tensor_i64_indice(%indices : tensor<4x1xi64>, %window: tensor<4x2x4xf32>, %data: tensor<16x8xf32>, %mask: tensor<4xi1>) -> tensor<16x8xf32> {
  %scatter = linalg_ext.scatter
              dimension_map = [1]
              ranged_data(true)
              overlap_window(false) signed_indice(true)
              ins(%window, %indices, %mask: tensor<4x2x4xf32>, tensor<4x1xi64>, tensor<4xi1>)
              outs(%data: tensor<16x8xf32>) {
                ^bb0(%arg0 :f32, %arg1: f32):
                  linalg_ext.yield %arg0 : f32
              } -> tensor<16x8xf32>
  return %scatter : tensor<16x8xf32>
}

// -----
// CHECK: linalg_ext.scatter
func.func @scatter_tensor_i16_indice(%indices : tensor<4x1xi16>, %window: tensor<4x2x4xf32>, %data: tensor<16x8xf32>, %mask: tensor<4xi1>) -> tensor<16x8xf32> {
  %scatter = linalg_ext.scatter
              dimension_map = [1]
              ranged_data(true)
              overlap_window(false) signed_indice(true)
              ins(%window, %indices, %mask: tensor<4x2x4xf32>, tensor<4x1xi16>, tensor<4xi1>)
              outs(%data: tensor<16x8xf32>) {
                ^bb0(%arg0 :f32, %arg1: f32):
                  linalg_ext.yield %arg0 : f32
              } -> tensor<16x8xf32>
  return %scatter : tensor<16x8xf32>
}

// -----
// CHECK: linalg_ext.scatter
func.func @scatter_nd_batch(
    %update : tensor<1x1x2x2xf32>, %indices : tensor<1x1x2xi32>,
    %init : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg_ext.scatter dimension_map = [0, 1]
    ranged_data(false)
    overlap_window(false) signed_indice(true)
      ins(%update, %indices : tensor<1x1x2x2xf32>, tensor<1x1x2xi32>)
      outs(%init : tensor<4x4xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}


// -----
// CHECK: linalg_ext.gather
func.func @gather_tensor(%A : tensor<4x1xi32>, %B: tensor<4x2x4xf32>, %C: tensor<16x8xf32>, %D: tensor<4xi1>) -> tensor<4x2x4xf32> {
  %gather = linalg_ext.gather
              dimension_map = [1]
              ranged_data(true) signed_indice(false)
              ins(%C, %A, %D: tensor<16x8xf32>, tensor<4x1xi32>, tensor<4xi1>)
              outs(%B: tensor<4x2x4xf32>) {
                ^bb0(%arg0 :f32, %arg1: f32):
                  linalg_ext.yield %arg0 : f32
              } -> tensor<4x2x4xf32>
  return %gather : tensor<4x2x4xf32>
}

// -----
// CHECK: linalg_ext.gather
func.func @gather_tensor_i8_indice(%indices : tensor<4x1xi8>, %window: tensor<4x2x4xf32>, %data: tensor<16x8xf32>, %mask: tensor<4xi1>) -> tensor<4x2x4xf32> {
  %gather = linalg_ext.gather
              dimension_map = [1]
              ranged_data(true) signed_indice(false)
              ins(%data, %indices, %mask: tensor<16x8xf32>, tensor<4x1xi8>, tensor<4xi1>)
              outs(%window: tensor<4x2x4xf32>) {
                ^bb0(%arg0 :f32, %arg1: f32):
                  linalg_ext.yield %arg0 : f32
              } -> tensor<4x2x4xf32>
  return %gather : tensor<4x2x4xf32>
}

// -----
func.func @gather_nd_batch(
    %init : tensor<1x1x2x2xf32>, %indices : tensor<1x1x2xi32>,
    %input : tensor<4x4xf32>) -> tensor<1x1x2x2xf32> {
  // expected-error @+1 {{indexed shape of init value dim#2 exceeds input value at dim#0 1 .vs. 4}}
  %0 = linalg_ext.gather dimension_map = [0, 1]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<4x4xf32>, tensor<1x1x2xi32>)
      outs(%init : tensor<1x1x2x2xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}

// -----
// CHECK: linalg_ext.gather_atomic_rmw
func.func @discrete_atomic_addf_with_mask(%arg0: memref<4x1xf32, 101>, %arg1: memref<4x1xi32, 101>, %arg2: memref<4xi8, 101>, %arg3: memref<?xf32, 1>, %arg4: memref<4x1xf32, 101>) {
  linalg_ext.gather_atomic_rmw addf relaxed ins(%arg0, %arg1, %arg2 : memref<4x1xf32, 101>, memref<4x1xi32, 101>, memref<4xi8, 101>) outs(%arg3, %arg4 : memref<?xf32, 1>, memref<4x1xf32, 101>)
  return
}

// -----
// CHECK: linalg_ext.atomic_rmw
func.func @atomic_contiguous(%alloc_5: memref<4xf32, 101>, %view_memref: memref<4xf32, 1>, %alloc_4: memref<4xf32, 101>) {
  linalg_ext.atomic_rmw addf release ins(%alloc_5 : memref<4xf32, 101>) outs(%view_memref, %alloc_4 : memref<4xf32, 1>, memref<4xf32, 101>) -> memref<4xf32, 1>, memref<4xf32, 101>
  return
}

// -----
// CHECK: linalg_ext.atomic_cas
func.func @atomic_cas(%arg0: tensor<128xi32>, %cmp: tensor<128xi32>, %val: tensor<128xi32>, %init: tensor<128xi32>) -> tensor<128xi32> {
  %0 = linalg_ext.atomic_cas relaxed ins(%arg0, %cmp, %val : tensor<128xi32>, tensor<128xi32>, tensor<128xi32>) outs(%init : tensor<128xi32>) -> tensor<128xi32>
  return %0: tensor<128xi32>
}

// -----
// CHECK: linalg_ext.gather_atomic_cas
func.func @gather_atomic_cas(%in: tensor<?xi32>, %cmp: tensor<128xi32>, %val: tensor<128xi32>, %indice: tensor<128xi64>, %init: tensor<128xi32>) -> tensor<128xi32> {
  %4 = linalg_ext.gather_atomic_cas release ins(%in, %cmp, %val, %indice: tensor<?xi32>, tensor<128xi32>, tensor<128xi32>, tensor<128xi64>) outs(%init : tensor<128xi32>) -> tensor<128xi32>
  return %4: tensor<128xi32>
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_tensor(%arg1: tensor<16x32x64xf32>, %arg2: tensor<16x32x64xf32>,
                               %init: tensor<16x32x64xf32>) {
  %libdevicecall = linalg_ext.libdevice_call
      ins(%arg1, %arg2 : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init: tensor<16x32x64xf32>)
      symbol = "__cn_vector_add_f32_rn" -> tensor<16x32x64xf32>
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_memref(%lhs: memref<16x32x64xi32, 101>, %rhs: memref<16x32x64xi32, 101>,
                               %init: memref<16x32x64xi32, 101>) {
  linalg_ext.libdevice_call
      ins(%lhs, %rhs : memref<16x32x64xi32, 101>, memref<16x32x64xi32, 101>)
      outs(%init: memref<16x32x64xi32, 101>)
      symbol = "__cn_vector_sub_s32"
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_init_output(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = tensor.empty () : tensor<128xf32>
  %libdevicecall = linalg_ext.libdevice_call
      ins(%arg0 : tensor<128xf32>)
      outs(%0 : tensor<128xf32>)
      symbol = "__cn_vector_abs_f32" -> tensor<128xf32>
  return %libdevicecall : tensor<128xf32>
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_sub(%lhs: memref<16x32x64xi32, 101>, %rhs: memref<16x32x64xi32, 101>,
                            %init: memref<16x32x64xi32, 101>) {
  linalg_ext.libdevice_call
      ins(%lhs, %rhs : memref<16x32x64xi32, 101>, memref<16x32x64xi32, 101>)
      outs(%init: memref<16x32x64xi32, 101>)
      symbol = "__cn_vector_sub_u32"
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_isinf(%lhs: memref<16x32x64xf32, 101>, %init: memref<16x32x64xi32, 101>) {
  linalg_ext.libdevice_call
      ins(%lhs : memref<16x32x64xf32, 101>)
      outs(%init: memref<16x32x64xi32, 101>)
      symbol = "__cn_vector_isinf_f32"
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_cast(%lhs: memref<16x32x64xf32, 101>, %init: memref<16x32x64xf16, 101>) {
  linalg_ext.libdevice_call
      ins(%lhs : memref<16x32x64xf32, 101>)
      outs(%init: memref<16x32x64xf16, 101>)
      symbol = "__cn_vector_cast_f32_to_f16_tz"
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_broadcast(%lhs: memref<16x32x64xi32, 101>, %rhs: i32,
                                  %init: memref<16x32x64xi32, 101>) {
  linalg_ext.libdevice_call
      ins(%lhs, %rhs : memref<16x32x64xi32, 101>, i32)
      outs(%init: memref<16x32x64xi32, 101>)
      symbol = "__cn_vector_max_scalar_s32"
  func.return
}

// -----
// CHECK: linalg_ext.libdevice_call
func.func @libdevice_call_eq(%lhs: tensor<16x32x64xf32>, %rhs: tensor<16x32x64xf32>,
                           %init: tensor<16x32x64xi8>) {
  %libdevicecall = linalg_ext.libdevice_call
      ins(%lhs, %rhs : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init: tensor<16x32x64xi8>)
      symbol = "__cn_vector_eq_f32" -> tensor<16x32x64xi8>
  func.return
}

// -----
func.func @libdevice_call_dynamic(%lhs: memref<16x?x64xi32, 101>, %rhs: memref<?x32x64xi32, 101>,
                                %init: memref<16x32x64xi32, 101>) {
  // CHECK: linalg_ext.libdevice_call
  linalg_ext.libdevice_call
      ins(%lhs, %rhs : memref<16x?x64xi32, 101>, memref<?x32x64xi32, 101>)
      outs(%init: memref<16x32x64xi32, 101>)
      symbol = "__cn_vector_sub_s32"
  func.return
}

// -----
func.func @libdevice_call_total_dynamic(%lhs: memref<?x?x?xi32, 101>, %rhs: memref<?x?x?xi32, 101>,
                                      %init: memref<?x?x?xi32, 101>) {
  // CHECK: linalg_ext.libdevice_call
  linalg_ext.libdevice_call
      ins(%lhs, %rhs : memref<?x?x?xi32, 101>, memref<?x?x?xi32, 101>)
      outs(%init: memref<?x?x?xi32, 101>)
      symbol = "__cn_vector_add_s32"
  func.return
}

// -----
func.func @scalar_libdevice_call_one_input(%arg0: f32) -> f32 {
  // CHECK: linalg_ext.scalar_libdevice_call
  %libdevicecall = linalg_ext.scalar_libdevice_call
      ins(%arg0 : f32)
      symbol = "__cn_scalar_abs_f32" -> f32
  return %libdevicecall : f32
}

// -----
func.func @scalar_libdevice_call_two_input(%arg0: f32) -> f32 {
  // CHECK: linalg_ext.scalar_libdevice_call
  %libdevicecall = linalg_ext.scalar_libdevice_call
      ins(%arg0, %arg0 : f32, f32)
      symbol = "__cn_scalar_add_f32" -> f32
  return %libdevicecall : f32
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_memref(%input : memref<4x4x16xf32>, %init : memref<6x8x16xf32>, %pvalue : f32) {
  linalg_ext.pad
      ins(%input : memref<4x4x16xf32>)
      outs(%init : memref<6x8x16xf32>)
      pvalue(%pvalue : f32)
      low = [1, 2, 0]
      high = [1, 2, 0] {
       ^bb0(%arg0 :index):
         linalg_ext.yield %arg0 : index
      }
  return
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_tensor(%input : tensor<4x4x16xf32>, %init : tensor<6x8x16xf32>, %pvalue : f32) -> tensor<6x8x16xf32> {
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4x16xf32>)
           outs(%init : tensor<6x8x16xf32>)
           pvalue(%pvalue : f32)
           low = [1, 2, 0]
           high = [1, 2, 0] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8x16xf32>
  return %pad : tensor<6x8x16xf32>
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_tensor_dynamic_low(%input : tensor<4x4x16xf32>, %init : tensor<6x?x16xf32>, %pvalue : f32, %arg : index) -> tensor<6x?x16xf32> {
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4x16xf32>)
           outs(%init : tensor<6x?x16xf32>)
           pvalue(%pvalue : f32)
           low = [1, %arg, 0]
           high = [1, 2, 0] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x?x16xf32>
  return %pad : tensor<6x?x16xf32>
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_tensor_dynamic_high(%input : tensor<4xf32>, %init : tensor<?xf32>, %pvalue : f32, %arg : index) -> tensor<?xf32> {
  %pad = linalg_ext.pad
           ins(%input : tensor<4xf32>)
           outs(%init : tensor<?xf32>)
           pvalue(%pvalue : f32)
           low = [1]
           high = [%arg] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<?xf32>
  return %pad : tensor<?xf32>
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_tensor_constant_pvalue(%input : tensor<4xf32>, %init : tensor<?xf32>, %arg : index) -> tensor<?xf32> {
  %cst = arith.constant 0.0 : f32
  %pad = linalg_ext.pad
           ins(%input : tensor<4xf32>)
           outs(%init : tensor<?xf32>)
           pvalue(%cst : f32)
           low = [1]
           high = [%arg] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<?xf32>
  return %pad : tensor<?xf32>
}

// -----
// CHECK: linalg_ext.pad
func.func @pad_tensor_dynamic_shape(%input : tensor<?x4xf32>, %init : tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %pad = linalg_ext.pad
           ins(%input : tensor<?x4xf32>)
           outs(%init : tensor<?x8xf32>)
           pvalue(%c0 : f32)
           low = [2, 2]
           high = [2, 2]{
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<?x8xf32>
  return %pad : tensor<?x8xf32>
}

// -----
// CHECK: linalg_ext.assert
func.func @assert_tensor(%arg0 : tensor<32xi32>) -> tensor<32xi32>{
  %1 = linalg_ext.assert {msg = "x > 0"} ins(%arg0 : tensor<32xi32>) -> tensor<32xi32>
  return %1 : tensor<32xi32>
}

// -----
// CHECK: linalg_ext.assert
func.func @assert_memref(%arg0 : memref<32xi32>) {
  linalg_ext.assert {msg = "x > 0"} ins(%arg0 : memref<32xi32>)
  return
}

// -----
// CHECK: linalg_ext.scan
func.func @scan_tensor(%input: tensor<16x32x64xf32>,
                       %output: tensor<16x32x64xf32>,
                       %init: tensor<16x64xf32>) {
  linalg_ext.scan
      ins(%input: tensor<16x32x64xf32>)
      outs(%output, %init: tensor<16x32x64xf32>, tensor<16x64xf32>)
      dimensions = [1]
      reverse = false
      {
      ^bb0(%in: f32, %out: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0: f32, f32
      } -> tensor<16x32x64xf32>, tensor<16x64xf32>
  func.return
}

// -----
// CHECK: linalg_ext.scan
func.func @scan_memref(%input: memref<16x32x64xf32>,
                       %output: memref<16x32x64xf32>,
                       %init: memref<16x64xf32>) {
  linalg_ext.scan
      ins(%input: memref<16x32x64xf32>)
      outs(%output, %init: memref<16x32x64xf32>, memref<16x64xf32>)
      dimensions = [1]
      reverse = false
      {
      ^bb0(%in: f32, %out: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0: f32, f32
      } -> memref<16x32x64xf32>, memref<16x64xf32>
  func.return
}

// -----
// CHECK: histogram_tensor
func.func @histogram_tensor(%input: tensor<128xi32>,
                            %init: tensor<16xi32>) -> tensor<16xi32> {
  %1 = linalg_ext.histogram
      ins(%input:tensor<128xi32>)
      outs(%init:tensor<16xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
}

// -----
// CHECK: histogram_memref
func.func @histogram_memref(%input: memref<128xi32>,
                            %init: memref<16xi32>) {
  linalg_ext.histogram
      ins(%input:memref<128xi32>)
      outs(%init:memref<16xi32>)
  func.return
}

// -----
// CHECK: linalg_ext.argmax
func.func @ext_argmax(%input_value: tensor<16x128xf32>, %input_index: tensor<16x128xi32>, %output_value: tensor<128xf32>, %output_index: tensor<128xi32>) {
  %argmax:2 = linalg_ext.argmax
                  ins(%input_value, %input_index : tensor<16x128xf32>, tensor<16x128xi32>)
                  outs(%output_value, %output_index : tensor<128xf32>, tensor<128xi32>)
                  dimensions = [0]
                  (%arg2: f32, %arg3: i32, %arg4: f32, %arg5: i32) {
                    %0 = arith.cmpf "oeq", %arg2, %arg4 : f32
                    %1 = arith.cmpi "slt", %arg3, %arg5 : i32
                    %2 = arith.andi %0, %1 : i1
                    %3 = arith.cmpf "ogt", %arg2, %arg4 : f32
                    %4 = arith.ori %3, %2 : i1
                    %5 = arith.select %4, %arg2, %arg4 : f32
                    %6 = arith.select %4, %arg3, %arg5 : i32
                    linalg.yield %5, %6 : f32, i32
                  }
  func.return
}

// -----
// CHECK: linalg_ext.argmin
func.func @ext_argmin(%input_value: tensor<16x128xf32>, %input_index: tensor<16x128xi32>, %output_value: tensor<128xf32>, %output_index: tensor<128xi32>) {
  %argmin:2 = linalg_ext.argmin
                  ins(%input_value, %input_index : tensor<16x128xf32>, tensor<16x128xi32>)
                  outs(%output_value, %output_index : tensor<128xf32>, tensor<128xi32>)
                  dimensions = [0]
                  (%arg2: f32, %arg3: i32, %arg4: f32, %arg5: i32) {
                    %0 = arith.cmpf "oeq", %arg2, %arg4 : f32
                    %1 = arith.cmpi "slt", %arg3, %arg5 : i32
                    %2 = arith.andi %0, %1 : i1
                    %3 = arith.cmpf "olt", %arg2, %arg4 : f32
                    %4 = arith.ori %3, %2 : i1
                    %5 = arith.select %4, %arg2, %arg4 : f32
                    %6 = arith.select %4, %arg3, %arg5 : i32
                    linalg.yield %5, %6 : f32, i32
                  }
  func.return
}
