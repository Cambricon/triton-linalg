// RUN: triton-linalg-opt %s -split-input-file -verify-diagnostics

func.func @transpose_invalid_permutation(%input: tensor<16x32x64xf32>,
    %output: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op permutation is not valid}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%output:tensor<32x64x16xf32>)
      permutation = [1, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----
func.func @transpose_permutated_dims_mismatch(%input: tensor<16x32x64xf32>,
    %output: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op dim(result, 0) = 32 doesn't match dim(input, permutation[0]) = 16}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%output:tensor<32x64x16xf32>)
      permutation = [0, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----
func.func @transpose_rank_permutation_size_mismatch(
    %input: tensor<16x32x64xf32>,
    %output: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op size of permutation 2 does not match the argument rank 3}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%output:tensor<32x64x16xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----
func.func @transpose_input_output_rank_mismatch(%input: tensor<16x32xf32>,
    %output: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op input rank 2 does not match init rank 3}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32xf32>)
      outs(%output:tensor<32x64x16xf32>)
      permutation = [1, 0, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----
func.func @reduce_unsorted_dim(%input: tensor<16x32x64xf32>,
                               %init: tensor<16xf32>) {
  // expected-error @+1 {{'linalg.reduce' op attribute 'dimensions' failed to satisfy constraint: i64 dense array attribute should be in increasing order}}
  linalg.reduce
      ins(%input: tensor<16x32x64xf32>)
      outs(%init: tensor<16xf32>)
      dimensions = [1, 0]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}

// -----
func.func @reduce_unmatched_dim(%input: tensor<16x32x64xf32>,
                                %init: tensor<16xf32>) {
  // expected-error @+1 {{'linalg.reduce' op number of dimensions after reduction 2 doesn't match the init rank 1}}
  linalg.reduce
      ins(%input: tensor<16x32x64xf32>)
      outs(%init: tensor<16xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}

// -----
func.func @reduce_out_of_range(%input: tensor<16x32x64xf32>,
                               %init: tensor<16xf32>) {
  // expected-error @+1 {{'linalg.reduce' op dimensions for reduction should be in the range [0, 2].}}
  linalg.reduce
      ins(%input: tensor<16x32x64xf32>)
      outs(%init: tensor<16xf32>)
      dimensions = [4]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return
}

// -----
func.func @broadcast_unmatched_dim(%input: tensor<16xf32>,
                                   %init: tensor<16x64xf32>) {
  // expected-error @+1 {{'linalg.broadcast' op input dim 0 should match init dim 1. input: 16, init: 64}}
  linalg.broadcast
      ins(%input: tensor<16xf32>)
      outs(%init: tensor<16x64xf32>)
      dimensions = [0]
  func.return
}

// -----
func.func @batch_conv_2d_nhwc_fhwc_invalid_shape_in_dilations(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.batch_conv_2d_nhwc_fhwc' op attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.batch_conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<2xi64>}
      ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----
func.func @batch_conv_2d_nhwc_fhwc_invalid_dtype_in_dilations(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.batch_conv_2d_nhwc_fhwc' op attribute 'dilations' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.batch_conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi32>, strides = dense<1> : tensor<2xi64>}
      ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----
func.func @batch_conv_2d_nhwc_fhwc_invalid_shape_in_strides(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.batch_conv_2d_nhwc_fhwc' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.batch_conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<1xi64>}
      ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----
func.func @batch_conv_2d_nhwc_fhwc_invalid_dtype_in_strides(%input: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>, %init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.batch_conv_2d_nhwc_fhwc' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.batch_conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi32>}
      ins (%input, %filter: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// -----
func.func @make_range_output_rank_invalid(%arg0: tensor<2x64xi32>) -> tensor<2x64xi32> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // expected-error @+1 {{'linalg_ext.make_range' op output rank must be 1}}
  %0 = linalg_ext.make_range ins(%c0, %c128 : i32, i32) outs(%arg0:tensor<2x64xi32>) -> tensor<2x64xi32>
  return %0 : tensor<2x64xi32>
}

// -----
func.func @make_range_start_end_invalid(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // expected-error @+1 {{'linalg_ext.make_range' op input argument end must greater than input arguments start in make range operation}}
  %0 = linalg_ext.make_range ins(%c128, %c0 : i32, i32) outs(%arg0:tensor<128xi32>) -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----
func.func @make_range_output_shape_mismatch(%arg0: tensor<129xi32>) -> tensor<129xi32> {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // expected-error @+1 {{'linalg_ext.make_range' op output shape mismatch}}
  %0 = linalg_ext.make_range ins(%c0, %c128 : i32, i32) outs(%arg0:tensor<129xi32>) -> tensor<129xi32>
  return %0 : tensor<129xi32>
}

// -----
func.func @make_range_result_type_invalid(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c2 = arith.constant 2 : i32
  %c130 = arith.constant 130 : i32
  // expected-error @+1 {{'linalg_ext.make_range' op result element type must be i32}}
  %0 = linalg_ext.make_range ins(%c2, %c130 : i32, i32) outs(%arg0:tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// -----
func.func @im2col_invalid_shape_in_strides(%input: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.im2col' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.im2col {strides = dense<1> : tensor<1xi64>}
      ins (%input: tensor<?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?x?xf32>
}

// -----
func.func @im2col_invalid_dtype_in_strides(%input: tensor<?x?x?x?xf32>, %init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32> {
  // expected-error @+1 {{'linalg_ext.im2col' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]}}
  %0 = linalg_ext.im2col {strides = dense<1> : tensor<2xi32>}
      ins (%input: tensor<?x?x?x?xf32>)
      outs (%init: tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?x?xf32>
}

// -----
func.func @scatter_extra_outputs(
    %update : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error @+1 {{expected the number of tensor results (2) to be equal to the number of output tensors (1)}}
  %0, %1 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<?x?x1xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----
func.func @scatter_mistmatch_dim_map_entries(
    %update : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{invalid number of dimension map entries}}
  %0 = linalg_ext.scatter dimension_map = [0, 1]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<?x?x1xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_nd_batch_exceed_dim(
    %update : tensor<1x1x8x8xf32>, %indices : tensor<1x1x2xi32>,
    %init : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{indexed shape of update value dim#2 exceeds init value at dim#0 8 .vs. 4}}
  %0 = linalg_ext.scatter dimension_map = [0, 1]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<1x1x8x8xf32>, tensor<1x1x2xi32>)
      outs(%init : tensor<4x4xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----
func.func @scatter_duplicate_dim_map_entries(
    %update : tensor<?x?x1xf32>, %indices : tensor<?x2xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = linalg_ext.scatter dimension_map = [1, 1]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<?x?x1xf32>, tensor<?x2xi32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_invalid_dim_map_entries(
    %update : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = linalg_ext.scatter dimension_map = [2]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<?x?x1xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_output_type_mismatch(
    %update : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<4x?xf32> {
  // expected-error @+1 {{expected type of operand #2 ('tensor<?x?xf32>') to match type of corresponding result ('tensor<4x?xf32>')}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
      ins(%update, %indices : tensor<?x?x1xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----
func.func @scatter_dim_mismatch(
    %update : tensor<?x?x1xf32>, %indices : tensor<48x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at batch dim}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xf32>, tensor<48x1xi32>)
    outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_dim_mismatch(
    %update : tensor<64x?x1xf32>, %indices : tensor<48x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at batch dim}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<64x?x1xf32>, tensor<48x1xi32>)
    outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_dim_mismatch(
    %update : tensor<?x?x?x?xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{op update value rank mismatch the rank of the init value}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x?x?xf32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
func.func @scatter_dim_mismatch(
    %update : tensor<?x1x4xf32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x3xf32>) -> tensor<?x3xf32> {
  // expected-error @+1 {{op indexed shape of update value dim#2 exceeds init value at dim#1}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x1x4xf32>, tensor<?x1xi32>)
    outs(%init : tensor<?x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----
func.func @scatter_region_type_mismatch(
    %update : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{expected region to have scalar argument of integer or float types}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi32>) {
    ^bb0(%arg1: index, %arg2: index):
      %1 = arith.addi %arg1, %arg2 : index
      %2 = arith.index_cast %1 : index to i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----
func.func @scatter_region_type_mismatch(
    %update : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 0 of region 'i64' and element type of update value 'i32'}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %1 = arith.trunci %arg1 : i64 to i32
      %2 = arith.addi %1, %arg2 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----
func.func @scatter_region_type_mismatch(
    %update : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 1 of region 'i64' and element type of init value 'i32'}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi32>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.trunci %arg2 : i64 to i32
      %2 = arith.addi %1, %arg1 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in region argument types 'i32' and 'i64'}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.extsi %arg1 : i32 to i64
      %2 = arith.addi %1, %arg2 : i64
      linalg_ext.yield %2 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_region_type_mismatch(
    %update : tensor<?x?x1xi64>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected region to have two arguments}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi64>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64, %arg3 : i64):
      %1 = arith.addi %arg1, %arg2 : i64
      linalg_ext.yield %1 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}


// -----
func.func @scatter_yield_mismatch(
    %update : tensor<?x?x1xi64>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi64>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{mismatch in type of yielded value 'i32' and argument of the region 'i64'}}
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_yield_mismatch(
    %update : tensor<?x?x1xi64>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi64>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{expected region to yield a single value}}
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_index_depth_dynamic(
    %update : tensor<?x?x1xi64>, %indices : tensor<?x?xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected index depth is static}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<?x?x1xi64>, tensor<?x?xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_init_rank_mismatch(
    %update : tensor<i64>, %indices : tensor<?x1xi32>,
    %init : tensor<i64>) -> tensor<i64> {
  // expected-error @+1 {{expected init value to be at least rank 1}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<i64>, tensor<?x1xi32>)
    outs(%init : tensor<i64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<i64>
  return %0 : tensor<i64>
}

// -----
func.func @scatter_init_rank_mismatch(
    %update : tensor<i64>, %indices : tensor<?x1xi32>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected update value to be at least rank 2}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices : tensor<i64>, tensor<?x1xi32>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_mask_shape_mismatch(
    %update : tensor<?x1x1xi64>, %indices : tensor<?x1xi32>, %mask : tensor<8xi1>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in shape of mask and update value at batch dim}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices, %mask : tensor<?x1x1xi64>, tensor<?x1xi32>, tensor<8xi1>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_mask_type_mismatch(
    %update : tensor<?x1x1xi64>, %indices : tensor<?x1xi32>, %mask : tensor<i16>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected mask to be of i1 element type and batch matched init}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices, %mask : tensor<?x1x1xi64>, tensor<?x1xi32>, tensor<i16>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @scatter_indice_type_mismatch(
    %update : tensor<?x1x1xi64>, %indices : tensor<?x1xi1>, %mask : tensor<i16>,
    %init : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected indices to be of rank 2 of i8/i16/i32/i64 element type}}
  %0 = linalg_ext.scatter dimension_map = [0]
    ranged_data(false) overlap_window(false) signed_indice(false)
    ins(%update, %indices, %mask : tensor<?x1x1xi64>, tensor<?x1xi1>, tensor<i16>)
    outs(%init : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----
func.func @gather_extra_outputs(
    %init : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error @+1 {{expected the number of tensor results (2) to be equal to the number of output tensors (1)}}
  %0, %1 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?x1xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----
func.func @gather_nd_batch_exceed_dim(
    %init : tensor<1x1x8x8xf32>, %indices : tensor<1x1x2xi32>,
    %input : tensor<4x4xf32>) -> tensor<1x1x8x8xf32> {
  // expected-error @+1 {{indexed shape of init value dim#2 exceeds input value at dim#0 8 .vs. 4}}
  %0 = linalg_ext.gather dimension_map = [0, 1]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<4x4xf32>, tensor<1x1x2xi32>)
      outs(%init : tensor<1x1x8x8xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<1x1x8x8xf32>
  return %0 : tensor<1x1x8x8xf32>
}

// -----
func.func @gather_mistmatch_dim_map_entries(
    %init : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xf32>) -> tensor<?x?x1xf32> {
  // expected-error @+1 {{invalid number of dimension map entries}}
  %0 = linalg_ext.gather dimension_map = [0, 1]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?x1xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?x1xf32>
  return %0 : tensor<?x?x1xf32>
}

// -----
func.func @gather_duplicate_dim_map_entries(
    %init : tensor<?x?x1xf32>, %indices : tensor<?x2xi32>,
    %input : tensor<?x?xf32>) -> tensor<?x?x1xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = linalg_ext.gather dimension_map = [1, 1]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<?x?xf32>, tensor<?x2xi32>)
      outs(%init : tensor<?x?x1xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?x1xf32>
  return %0 : tensor<?x?x1xf32>
}

// -----
func.func @gather_invalid_dim_map_entries(
    %init : tensor<?x?x1xf32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xf32>) -> tensor<?x?x1xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = linalg_ext.gather dimension_map = [2]
    ranged_data(false) signed_indice(false)
      ins(%input, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%init : tensor<?x?x1xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?x1xf32>
  return %0 : tensor<?x?x1xf32>
}

// -----
func.func @gather_dim_mismatch(
    %init : tensor<?x?x1xf32>, %indices : tensor<48x1xi32>,
    %input : tensor<?x?xf32>) -> tensor<?x?x1xf32> {
  // expected-error @+1 {{mismatch in shape of indices and init value at batch dim}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xf32>, tensor<48x1xi32>)
    outs(%init : tensor<?x?x1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?x1xf32>
  return %0 : tensor<?x?x1xf32>
}

// -----
func.func @gather_dim_mismatch(
    %init : tensor<?x?x?x?xf32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{op init value rank exceeds the rank of the input value}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----
func.func @gather_dim_mismatch(
    %init : tensor<?x1x4xf32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x3xf32>) -> tensor<?x1x4xf32> {
  // expected-error @+1 {{op indexed shape of init value dim#2 exceeds input value at dim#1}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x3xf32>, tensor<?x1xi32>)
    outs(%init : tensor<?x1x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x1x4xf32>
  return %0 : tensor<?x1x4xf32>
}

// -----
func.func @gather_region_type_mismatch(
    %init : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xi32>) -> tensor<?x?x1xi32> {
  // expected-error @+1 {{expected region to have scalar argument of integer or float types}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?x1xi32>) {
    ^bb0(%arg1: index, %arg2: index):
      %1 = arith.addi %arg1, %arg2 : index
      %2 = arith.index_cast %1 : index to i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?x1xi32>
  return %0 : tensor<?x?x1xi32>
}

// -----
func.func @gather_region_type_mismatch(
    %init : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xi32>) -> tensor<?x?x1xi32> {
  // expected-error @+1 {{mismatch in argument 0 of region 'i64' and element type of init value 'i32'}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?x1xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %1 = arith.trunci %arg1 : i64 to i32
      %2 = arith.addi %1, %arg2 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?x1xi32>
  return %0 : tensor<?x?x1xi32>
}

// -----
func.func @gather_region_type_mismatch(
    %init : tensor<?x?x1xi32>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xi32>) -> tensor<?x?x1xi32> {
  // expected-error @+1 {{mismatch in argument 1 of region 'i64' and element type of input value 'i32'}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%init : tensor<?x?x1xi32>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.trunci %arg2 : i64 to i32
      %2 = arith.addi %1, %arg1 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?x1xi32>
  return %0 : tensor<?x?x1xi32>
}

// -----
func.func @gather_yield_mismatch(
    %init : tensor<?x?x1xi64>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xi64>) -> tensor<?x?x1xi64> {
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%init : tensor<?x?x1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{expected region to yield a single value}}
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?x1xi64>
  return %0 : tensor<?x?x1xi64>
}

// -----
func.func @gather_index_depth_dynamic(
    %init : tensor<?x?x1xi64>, %indices : tensor<?x?xi32>,
    %input : tensor<?x?xi64>) -> tensor<?x?x1xi64> {
  // expected-error @+1 {{expected index depth is static}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi64>, tensor<?x?xi32>)
    outs(%init : tensor<?x?x1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?x1xi64>
  return %0 : tensor<?x?x1xi64>
}

// -----
func.func @gather_input_rank_mismatch(
    %init : tensor<i64>, %indices : tensor<?x1xi32>,
    %input : tensor<i64>) -> tensor<i64> {
  // expected-error @+1 {{expected input value to be at least rank 1}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<i64>, tensor<?x1xi32>)
    outs(%init : tensor<i64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<i64>
  return %0 : tensor<i64>
}

// -----
func.func @gather_input_rank_mismatch(
    %init : tensor<i64>, %indices : tensor<?x1xi32>,
    %input : tensor<?x?xi64>) -> tensor<i64> {
  // expected-error @+1 {{expected init value to be at least rank 2}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%init : tensor<i64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<i64>
  return %0 : tensor<i64>
}

// -----
func.func @gather_mask_shape_mismatch(
    %init : tensor<?x1x1xi64>, %indices : tensor<?x1xi32>, %mask : tensor<8xi1>,
    %input : tensor<?x?xi64>) -> tensor<?x1x1xi64> {
  // expected-error @+1 {{mismatch in shape of mask and init value at batch dim}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices, %mask : tensor<?x?xi64>, tensor<?x1xi32>, tensor<8xi1>)
    outs(%init : tensor<?x1x1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x1x1xi64>
  return %0 : tensor<?x1x1xi64>
}

// -----
func.func @gather_mask_type_mismatch(
    %init : tensor<?x1x1xi64>, %indices : tensor<?x1xi32>, %mask : tensor<i16>,
    %input : tensor<?x?xi64>) -> tensor<?x1x1xi64> {
  // expected-error @+1 {{expected mask to be of i1 element type and batch matched init}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices, %mask : tensor<?x?xi64>, tensor<?x1xi32>, tensor<i16>)
    outs(%init : tensor<?x1x1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x1x1xi64>
  return %0 : tensor<?x1x1xi64>
}

// -----
func.func @gather_indice_type_mismatch(
    %init : tensor<?x1x1xi64>, %indices : tensor<?x1xi1>, %mask : tensor<i16>,
    %input : tensor<?x?xi64>) -> tensor<?x1x1xi64> {
  // expected-error @+1 {{expected indices to be of rank 2 of i8/i16/i32/i64 element type}}
  %0 = linalg_ext.gather dimension_map = [0]
    ranged_data(false) signed_indice(false)
    ins(%input, %indices, %mask : tensor<?x?xi64>, tensor<?x1xi1>, tensor<i16>)
    outs(%init : tensor<?x1x1xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x1x1xi64>
  return %0 : tensor<?x1x1xi64>
}

// -----
func.func @pad_rank_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8x16xf32>, %pvalue : f32) -> tensor<6x8x16xf32> {
  // expected-error @+1 {{'linalg_ext.pad' op expected same rank of input and init}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8x16xf32>)
           pvalue(%pvalue : f32)
           low = [1, 2]
           high = [1, 2] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8x16xf32>
  return %pad : tensor<6x8x16xf32>
}

// -----
func.func @pad_static_low_rank_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8xf32>, %pvalue : f32) -> tensor<6x8xf32> {
  // expected-error @+1 {{'linalg_ext.pad' op expected same size of static_low and input}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8xf32>)
           pvalue(%pvalue : f32)
           low = [1]
           high = [1, 2] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8xf32>
  return %pad : tensor<6x8xf32>
}

// -----
func.func @pad_static_high_rank_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8xf32>, %pvalue : f32) -> tensor<6x8xf32> {
  // expected-error @+1 {{'linalg_ext.pad' op expected same size of static_high and input}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8xf32>)
           pvalue(%pvalue : f32)
           low = [1, 2]
           high = [1, 2, 3] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8xf32>
  return %pad : tensor<6x8xf32>
}

// -----
func.func @pad_shape_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8xf32>, %pvalue : f32) -> tensor<6x8xf32> {
  // expected-error @+1 {{specified type 'tensor<6x8xf32>' does not match the inferred type 'tensor<6x7xf32>' on dimension 1}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8xf32>)
           pvalue(%pvalue : f32)
           low = [1, 2]
           high = [1, 1] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8xf32>
  return %pad : tensor<6x8xf32>
}

// -----
func.func @pad_type_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8xf16>, %pvalue : f32) -> tensor<6x8xf16> {
  // expected-error @+1 {{'linalg_ext.pad' op expected same element type of input and init}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8xf16>)
           pvalue(%pvalue : f32)
           low = [1, 2]
           high = [1, 2] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8xf16>
  return %pad : tensor<6x8xf16>
}

// -----
func.func @pad_pvalue_type_mismatch(%input : tensor<4x4xf32>, %init : tensor<6x8xf32>, %pvalue : f16) -> tensor<6x8xf32> {
  // expected-error @+1 {{'linalg_ext.pad' op expected same type of padding value and input elements}}
  %pad = linalg_ext.pad
           ins(%input : tensor<4x4xf32>)
           outs(%init : tensor<6x8xf32>)
           pvalue(%pvalue : f16)
           low = [1, 2]
           high = [1, 2] {
            ^bb0(%arg0 :index):
              linalg_ext.yield %arg0 : index
           } -> tensor<6x8xf32>
  return %pad : tensor<6x8xf32>
}

// -----
func.func @scan_unmatched_output_and_init_num(%input: tensor<16x32x64xf32>,
                                              %output0: tensor<16x32x64xf32>,
                                              %output1: tensor<16x32x64xf32>,
                                              %init: tensor<16x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects outputs paired with inits. but sum of outputs and inits are 3.}}
  linalg_ext.scan
      ins(%input: tensor<16x32x64xf32>)
      outs(%output0, %output1, %init: tensor<16x32x64xf32>, tensor<16x32x64xf32>, tensor<16x64xf32>)
      dimensions = [1]
      reverse = false
      {
      ^bb0(%in: f32, %out0: f32, %out1: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0, %0: f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_dim(%input: tensor<16x32x64xf32>,
                              %output: tensor<16x64xf32>,
                              %init: tensor<16xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects inputs and outputs have the same shapes. Shape at input-index 0 is not equal to the shape at output-index 0.}}
  linalg_ext.scan
      ins(%input: tensor<16x32x64xf32>)
      outs(%output, %init: tensor<16x64xf32>, tensor<16xf32>)
      dimensions = [1]
      reverse = false
      {
      ^bb0(%in: f32, %out: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0: f32, f32
      }
  func.return
}

// -----
func.func @scan_out_of_range(%input: tensor<16x32x64xf32>,
                             %output: tensor<16x32x64xf32>,
                             %init: tensor<32x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op dimension for scan should be in the range [0, 2].}}
  linalg_ext.scan
      ins(%input: tensor<16x32x64xf32>)
      outs(%output, %init: tensor<16x32x64xf32>, tensor<32x64xf32>)
      dimensions = [4]
      reverse = false
      {
      ^bb0(%in: f32, %out: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0: f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_input_and_output_shape(%input0: tensor<16x32x64xf32>,
                                                 %input1: tensor<16x32x64xf32>,
                                                 %output0: tensor<32x64xf32>,
                                                 %output1: tensor<32x64xf32>,
                                                 %init0: tensor<32x64xf32>,
                                                 %init1: tensor<32x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects inputs and outputs have the same shapes. Shape at input-index 0 is not equal to the shape at output-index 0.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_inputs_shape(%input0: tensor<16x32x64xf32>,
                                       %input1: tensor<32x64xf32>,
                                       %output0: tensor<16x32x64xf32>,
                                       %output1: tensor<16x32x64xf32>,
                                       %init0: tensor<32x64xf32>,
                                       %init1: tensor<32x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects all inputs have the same shapes. Shape at input-index 1 is not equal to the shape at input-index 0.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<16x32x64xf32>, tensor<16x32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_outputs_shape(%input0: tensor<16x32x64xf32>,
                                        %input1: tensor<16x32x64xf32>,
                                        %output0: tensor<16x32x64xf32>,
                                        %output1: tensor<32x64xf32>,
                                        %init0: tensor<32x64xf32>,
                                        %init1: tensor<32x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects all outputs have the same shapes. Shape at output-index 1 is not equal to the shape at output-index 0.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<16x32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_inits_shape(%input0: tensor<16x32x64xf32>,
                                      %input1: tensor<16x32x64xf32>,
                                      %output0: tensor<16x32x64xf32>,
                                      %output1: tensor<16x32x64xf32>,
                                      %init0: tensor<32x64xf32>,
                                      %init1: tensor<16x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op expects all inits have the same shapes. Shape at init-index 3 is not equal to the shape at init-index 2.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<16x32x64xf32>, tensor<16x32x64xf32>, tensor<32x64xf32>, tensor<16x64xf32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unexpected_inits_shape(%input0: tensor<16x32x64xf32>,
                                       %input1: tensor<16x32x64xf32>,
                                       %output0: tensor<16x32x64xf32>,
                                       %output1: tensor<16x32x64xf32>,
                                       %init0: tensor<16x64xf32>,
                                       %init1: tensor<16x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op inits shape is not equal to the expected shape 32, 64.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<16x32x64xf32>, tensor<16x32x64xf32>, tensor<16x64xf32>, tensor<16x64xf32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_block_args_num(%input: tensor<16xf32>,
                                         %output: tensor<16xf32>,
                                         %init: tensor<f32>) {
  // expected-error @+1 {{'linalg_ext.scan' op mismatching number of operands and block arguments}}
  linalg_ext.scan
      ins(%input: tensor<16xf32>)
      outs(%output, %init: tensor<16xf32>, tensor<f32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in: f32, %out0: f32,  %out1: f32, %ini: f32):
        %0 = arith.addf %ini, %in: f32
        linalg_ext.yield %0, %0: f32, f32
      }
  func.return
}

// -----
func.func @scan_unmatched_element_type(%input: tensor<16xf32>,
                                       %output: tensor<16xf32>,
                                       %init: tensor<f32>) {
  // expected-error @+1 {{'linalg_ext.scan' op input element type 'f32' does not match corresponding block argument type 'i32'}}
  linalg_ext.scan
      ins(%input: tensor<16xf32>)
      outs(%output, %init: tensor<16xf32>, tensor<f32>)
      dimensions = [0]
      reverse = false
      {
      ^bb0(%in: i32, %out: i32, %ini: i32):
        %0 = arith.addi %ini, %in: i32
        linalg_ext.yield %0, %0: i32, i32
      }
  func.return
}

// -----
func.func @scan_multi_operands(%input0: tensor<16x32x64xf32>,
                               %input1: tensor<16x32x64xf32>,
                               %output0: tensor<16x32x64xf32>,
                               %output1: tensor<16x32x64xf32>,
                               %init0: tensor<32x64xf32>,
                               %init1: tensor<32x64xf32>) {
  // expected-error @+1 {{'linalg_ext.scan' op only support single dimension.}}
  linalg_ext.scan
      ins(%input0, %input1: tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%output0, %output1, %init0, %init1: tensor<16x32x64xf32>, tensor<16x32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>)
      dimensions = [0, 1]
      reverse = false
      {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32, %ini0: f32, %ini1: f32):
        %0 = arith.addf %ini0, %in0: f32
        %1 = arith.subf %ini1, %in1: f32
        linalg_ext.yield %0, %1, %0, %1: f32, f32, f32, f32
      }
  func.return
}

// -----
func.func @scalar_libdevice_call_input_invalid(%arg0: tensor<f32>) -> f32 {
  // expected-error @+1 {{'linalg_ext.scalar_libdevice_call' op expects all input types are scalar type.}}
  %libdevicecall = linalg_ext.scalar_libdevice_call
      ins(%arg0 : tensor<f32>)
      symbol = "__cn_scalar_abs_f32" -> f32
  return %libdevicecall : f32
}

// -----
func.func @scalar_libdevice_call_result_invalid(%arg0: f32) -> tensor<f32> {
  // expected-error @+1 {{'linalg_ext.scalar_libdevice_call' op expects the result type is scalar type.}}
  %libdevicecall = linalg_ext.scalar_libdevice_call
      ins(%arg0, %arg0 : f32, f32)
      symbol = "__cn_scalar_add_f32" -> tensor<f32>
  return %libdevicecall : tensor<f32>
}
