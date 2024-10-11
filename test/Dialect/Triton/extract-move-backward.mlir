// RUN: triton-linalg-opt --extract-like-move-backward --canonicalize %s -split-input-file | FileCheck %s

// CHECK-LABEL: @extract_element_from_fill
func.func @extract_element_from_fill(%arg0: i64) -> i64 {
  // CHECK: return %arg0 : i64
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<128xi64>
  %1 = linalg.fill ins(%arg0 : i64) outs(%0 : tensor<128xi64>) -> tensor<128xi64>
  %2 = tensor.extract %1[%c0] : tensor<128xi64>
  return %2 : i64
}

// -----
// CHECK-LABEL: @extract_element_from_make_range
func.func @extract_element_from_make_range(%arg0: i64) -> i32 {
  // CHECK: %[[C2_I32:.*]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[ARG_CAST_INDEX:.*]] = arith.index_cast %arg0 : i64 to index
  // CHECK-NEXT: %[[ARG_CAST_I32:.*]] = arith.index_cast %[[ARG_CAST_INDEX]] : index to i32
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG_CAST_I32]], %[[C2_I32]]
  // CHECK-NOT: linalg.*
  %c2 = arith.constant 2 : i32
  %c130 = arith.constant 130 : i32
  %0 = tensor.empty() : tensor<128xi32>
  %1 = linalg_ext.make_range ins(%c2, %c130 : i32, i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
  %offset = arith.index_cast %arg0 : i64 to index
  %2 = tensor.extract %1[%offset] : tensor<128xi32>
  return %2 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_arith_addi
func.func @extract_element_from_arith_addi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : i32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.addi %arg0, %arg1 : tensor<128xi32>
  %1 = tensor.extract %0[%c0] : tensor<128xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_math_erf
func.func @extract_element_from_math_erf(%arg0: tensor<128xf32>) -> f32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[ERF:.*]] = math.erf %[[ARG_EXTRA]] : f32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = math.erf %arg0 : tensor<128xf32>
  %1 = tensor.extract %0[%c0] : tensor<128xf32>
  return %1 : f32
}

// -----
// CHECK-LABEL: @extract_element_from_arith_addf
func.func @extract_element_from_arith_addf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> f32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : f32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32>
  %1 = tensor.extract %0[%c0] : tensor<128xf32>
  return %1 : f32
}

// -----
// CHECK-LABEL: @extract_element_from_arith_cmpi
func.func @extract_element_from_arith_cmpi(%arg0: tensor<128xindex>, %arg1: tensor<128xindex>) -> i1 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<128xindex>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xindex>
  // CHECK-NEXT: %[[CMPI:.*]] = arith.cmpi slt, %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : index
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.cmpi slt, %arg0, %arg1 : tensor<128xindex>
  %1 = tensor.extract %0[%c0] : tensor<128xi1>
  return %1 : i1
}

// -----
// CHECK-LABEL: @extract_element_from_arith_cmpf
func.func @extract_element_from_arith_cmpf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> i1 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[CMPF:.*]] = arith.cmpf olt, %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : f32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<128xf32>
  %1 = tensor.extract %0[%c0] : tensor<128xi1>
  return %1 : i1
}

// -----
// CHECK-LABEL: @extract_element_from_arith_bitcast
func.func @extract_element_from_arith_bitcast(%arg0: tensor<128xi32>) -> f32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[BITCAST:.*]] = arith.bitcast %[[ARG_EXTRA0]] : i32 to f32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.bitcast %arg0 : tensor<128xi32> to tensor<128xf32>
  %1 = tensor.extract %0[%c0] : tensor<128xf32>
  return %1 : f32
}

// -----
// CHECK-LABEL: @extract_element_from_arith_negf
func.func @extract_element_from_arith_negf(%arg0: tensor<128xf32>) -> f32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[NEGF:.*]] = arith.negf %[[ARG_EXTRA0]] : f32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.negf %arg0 : tensor<128xf32>
  %1 = tensor.extract %0[%c0] : tensor<128xf32>
  return %1 : f32
}

// -----
// CHECK-LABEL: @extract_element_from_arith_trunci
func.func @extract_element_from_arith_trunci(%arg0: tensor<128xi32>) -> i16 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[TRUNCI:.*]] = arith.trunci %[[ARG_EXTRA0]] : i32 to i16
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.trunci %arg0 : tensor<128xi32> to tensor<128xi16>
  %1 = tensor.extract %0[%c0] : tensor<128xi16>
  return %1 : i16
}

// -----
// CHECK-LABEL: @extract_element_from_arith_truncf
func.func @extract_element_from_arith_truncf(%arg0: tensor<128xf32>) -> bf16 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xf32>
  // CHECK-NEXT: %[[TRUNCF:.*]] = arith.truncf %[[ARG_EXTRA0]] : f32 to bf16
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.truncf %arg0 : tensor<128xf32> to tensor<128xbf16>
  %1 = tensor.extract %0[%c0] : tensor<128xbf16>
  return %1 : bf16
}

// -----
// CHECK-LABEL: @extract_element_from_arith_and
func.func @extract_element_from_arith_and(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NEXT: %[[AND:.*]] = arith.andi %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : i32
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = arith.andi %arg0, %arg1 : tensor<128xi32>
  %1 = tensor.extract %0[%c0] : tensor<128xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_map_arith_add_with_two_indices
func.func @extract_element_from_map_arith_add_with_two_indices(
    %arg0: tensor<128x16xi32>, %arg1: tensor<128x16xi32>) -> i32 {
  // CHECK:      %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]], %[[C0_INDEX]]] : tensor<128x16xi32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]], %[[C0_INDEX]]] : tensor<128x16xi32>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : i32
  // CHECK-NOT:  linalg.*
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<128x16xi32>
  %1 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128x16xi32>, tensor<128x16xi32>) outs(%0 : tensor<128x16xi32>)
  %2 = tensor.extract %1[%c0, %c0] : tensor<128x16xi32>
  return %2 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_map_with_two_payloads
func.func @extract_element_from_map_with_two_payloads(%arg0: tensor<32xi64>, %arg1: tensor<32xi32>) -> i64 {
  // CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  // CHECK: %[[C0_INDEX:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i64
  // CHECK: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[%[[C0_INDEX]]] : tensor<32xi32>
  // CHECK: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<32xi64>
  // CHECK: %[[EXT:.*]] = arith.extsi %[[ARG_EXTRA0]] : i32 to i64
  // CHECK: %[[ADD_0:.*]] = arith.addi %[[ARG_EXTRA1]], %[[EXT]] : i64
  // CHECK: %[[ADD_1:.*]] = arith.addi %[[ADD_0]], %[[C1_I64]] : i64
  %0 = tensor.empty() : tensor<32xi64>
  %1 = linalg.map ins(%arg0, %arg1 : tensor<32xi64>, tensor<32xi32>) outs(%0 : tensor<32xi64>)
  (%in: i64, %in1:i32) {
    %2 = arith.extsi %in1 : i32 to i64
    %3 = arith.addi %in, %2 : i64
    %4 = arith.addi %3, %c1 : i64
    linalg.yield %4 : i64
  }
  %5 = tensor.extract %1[%c0] : tensor<32xi64>
  return %5 : i64
}

// -----
// CHECK-LABEL: @extract_element_from_broadcast_op
func.func @extract_element_from_broadcast_op(%arg0: tensor<128xi32>) -> i32 {
  // CHECK:      %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0_INDEX]]] : tensor<128xi32>
  // CHECK-NOT:  linalg.broadcast
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %1 = tensor.empty() : tensor<128x16xi32>
  %2 = linalg.broadcast ins(%arg0: tensor<128xi32>) outs(%1: tensor<128x16xi32>) dimensions = [1]
  %3 = tensor.extract %2[%c0, %c5] : tensor<128x16xi32>
  return %3 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_collapse_shape_op_with_static_shape
func.func @extract_element_from_collapse_shape_op_with_static_shape(%arg0: tensor<128x16x4xi32>, %arg1: index) -> i32 {
  // CHECK-DAG: %[[C1_INDEX:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2_INDEX:.*]] = arith.constant 2 : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %arg0[%arg1, %[[C2_INDEX]], %[[C1_INDEX]]] : tensor<128x16x4xi32>
  // CHECK-NOT: tensor.collapse_shape
  %c9 = arith.constant 9 : index
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<128x16x4xi32> into tensor<128x64xi32>
  %1 = tensor.extract %0[%arg1, %c9] : tensor<128x64xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_collapse_shape_op_with_dynamic_shape
func.func @extract_element_from_collapse_shape_op_with_dynamic_shape(%arg0: tensor<?x?x?xi32>) -> i32 {
  // CHECK-DAG: %[[C2_INDEX:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C1_INDEX:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C9_INDEX:.*]] = arith.constant 9 : index
  // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %arg0, %[[C2_INDEX]] : tensor<?x?x?xi32>
  // CHECK-DAG: %[[VAL1:.*]] = arith.remui %[[C9_INDEX]], %[[DIM0]] : index
  // CHECK-DAG: %[[VAL2:.*]] = arith.divui %[[C9_INDEX]], %[[DIM0]] : index
  // CHECK-DAG: %[[VAL3:.*]] = tensor.dim %arg0, %[[C1_INDEX]] : tensor<?x?x?xi32>
  // CHECK-DAG: %[[VAL4:.*]] = arith.remui %[[VAL2]], %[[VAL3]] : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C2_INDEX]], %[[VAL4]], %[[VAL1]]] : tensor<?x?x?xi32>
  // CHECK-NOT: tensor.collapse_shape
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x?x?xi32> into tensor<?x?xi32>
  %1 = tensor.extract %0[%c2, %c9] : tensor<?x?xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_expand_shape_op_with_src_dimension_size_1
func.func @extract_element_from_expand_shape_op_with_src_dimension_size_1(%arg0: tensor<2x12xi32>) -> i32 {
  // CHECK-DAG: %[[C11_INDEX:.*]] = arith.constant 11 : index
  // CHECK-DAG: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0_INDEX]], %[[C11_INDEX]]] : tensor<2x12xi32>
  // CHECK-NOT: tensor.expand_shape
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [2, 3, 4] : tensor<2x12xi32> into tensor<2x3x4xi32>
  %1 = tensor.extract %0[%c0, %c2, %c3] : tensor<2x3x4xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @extract_element_from_expand_shape_op_with_static_shape
func.func @extract_element_from_expand_shape_op_with_static_shape(%arg0: tensor<128x64xi32>, %arg1: index) -> i32 {
  // CHECK-DAG: %[[C64_INDEX:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[C58_INDEX:.*]] = arith.constant 58 : index
  // CHECK-DAG: %[[ADD:.*]] = arith.addi %arg1, %[[C64_INDEX]] : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %arg0[%[[ADD]], %[[C58_INDEX]]] : tensor<128x64xi32>
  // CHECK-NOT: tensor.collapse_shape
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [4, 32, 4, 16] : tensor<128x64xi32> into tensor<4x32x4x16xi32>
  %1 = tensor.extract %0[%c2, %arg1, %c3, %c10] : tensor<4x32x4x16xi32>
  return %1 : i32
}

// -----
// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 floordiv 6)>
// CHECK-LABEL: @extract_element_from_expand_shape_op_with_dynamic_shape
func.func @extract_element_from_expand_shape_op_with_dynamic_shape(%arg0: tensor<?x?xi32>) -> i32 {
  // CHECK-DAG: %[[C1_INDEX:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C22_INDEX:.*]] = arith.constant 22 : index
  // CHECK-DAG: %[[C3_INDEX:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[DIM:.*]] = tensor.dim %arg0, %[[C1_INDEX]] : tensor<?x?xi32>
  // CHECK-DAG: %[[AFFINE:.*]] = affine.apply #[[MAP]]()[%[[DIM]]]
  // CHECK-DAG: %[[VAL0:.*]] = arith.muli %[[AFFINE]], %[[C3_INDEX]] : index
  // CHECK-DAG: %[[VAL1:.*]] = arith.addi %[[VAL0]], %[[C3_INDEX]] : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C22_INDEX]], %[[VAL1]]] : tensor<?x?xi32>
  // CHECK-NOT: tensor.collapse_shape
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c10 = arith.constant 10 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %new_dim0 = arith.divui %dim0, %c10 : index
  %new_dim1 = arith.divui %dim1, %c6 : index
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [%new_dim0, 10, 6, %new_dim1] : tensor<?x?xi32> into tensor<?x10x6x?xi32>
  %1 = tensor.extract %0[%c2, %c2, %c3, %c3] : tensor<?x10x6x?xi32>
  return %1 : i32
}

// -----
// CHECK-LABEL:   func.func @extract_element_from_constant_with_splat_attr() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
func.func @extract_element_from_constant_with_splat_attr() -> i32 {
  %cst = arith.constant dense<0> : tensor<16xi32>
  %c5 = arith.constant 1 : index
  %extracted = tensor.extract %cst[%c5] : tensor<16xi32>
  return %extracted : i32
}

// -----
// CHECK-LABEL:   func.func @extract_element_from_constant_with_array_attr() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
func.func @extract_element_from_constant_with_array_attr() -> i32 {
  %cst = arith.constant dense<[2, 1, 3, 4, 5, 1, 2, 3]> : tensor<8xi32>
  %c4 = arith.constant 4 : index
  %extracted = tensor.extract %cst[%c4] : tensor<8xi32>
  return %extracted : i32
}

// -----
func.func @extract_from_for_iter_args(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
    %1 = tensor.extract %arg6[%c0, %c0] : tensor<64x64xi64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [64, %2], strides: [1, 1] : !llvm.ptr<1> to memref<64x?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<64x?xf32, 1>
    %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %mapped = linalg.map { math.absf } ins(%4 : tensor<64x64xf32>) outs(%arg5 : tensor<64x64xf32>)
    %5 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%5 : tensor<64x64xi64>)
    scf.yield %mapped, %mapped_0 : tensor<64x64xf32>, tensor<64x64xi64>
  }
  return %0#0 : tensor<64x64xf32>
}
// CHECK-LABEL: @extract_from_for_iter_args
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[PTR:.*]] = llvm.inttoptr %[[ARG0:.*]] : i64 to !llvm.ptr<1>
// CHECK-DAG:   %[[EXTRACT0:.*]] = tensor.extract %[[ARG3:.*]][%[[C0]], %[[C0]]] : tensor<64x64xi64>
// CHECK-DAG:   %[[EXTRACT1:.*]] = tensor.extract %[[ARG2:.*]][%[[C0]], %[[C0]]] : tensor<64x64xi64>
// CHECK:       %[[VAL1:.*]]:2 = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[ARG1:.*]], %[[ARG6:.*]] = %[[EXTRACT1]]) -> (tensor<64x64xf32>, i64) {
// CHECK:         %[[ADDI:.*]] = arith.addi %[[ARG6]], %[[EXTRACT0]] : i64
// CHECK:         %[[VAL2:.*]] = arith.index_cast %[[ARG6]] : i64 to index
// CHECK:         %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [64, %[[VAL2]]]
// CHECK:         %[[VAL3:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:         %[[VAL4:.*]] = tensor.extract_slice %[[VAL3]][%[[VAL2]], %[[VAL2]]] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
// CHECK:         %[[VAL_MAP:.*]] = linalg.map { math.absf } ins(%[[VAL4]] : tensor<64x64xf32>) outs(%[[ARG5]] : tensor<64x64xf32>)
// CHECK:         scf.yield %[[VAL_MAP]], %[[ADDI]] : tensor<64x64xf32>, i64
// CHECK:       }
// CHECK:       return %[[VAL1]]#0 : tensor<64x64xf32>

// -----
func.func @extract_from_for_iter_args_with_few_compute(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>, %arg4: index, %arg5: index) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %arg1, %arg8 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
    %1 = arith.addi %arg4, %arg5 : index
    %2 = arith.muli %arg4, %1 : index
    %3 = arith.subi %2, %arg5 : index
    %extracted = tensor.extract %arg8[%1, %3] : tensor<64x64xi64>
    %4 = arith.index_cast %extracted : i64 to index
    %5 = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %view_memref = aux.view %5 to offset: [0], sizes: [64, %4], strides: [1, 1] : <1> to memref<64x?xf32, 1>
    %6 = bufferization.to_tensor %view_memref : memref<64x?xf32, 1>
    %extracted_slice = tensor.extract_slice %6[%4, %4] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %mapped = linalg.map { math.absf } ins(%extracted_slice : tensor<64x64xf32>) outs(%arg7 : tensor<64x64xf32>)
    %7 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi } ins(%arg8, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%7 : tensor<64x64xi64>)
    scf.yield %mapped, %mapped_0 : tensor<64x64xf32>, tensor<64x64xi64>
  }
  return %0#0 : tensor<64x64xf32>
}
// CHECK-LABEL: @extract_from_for_iter_args_with_few_compute
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[T0:.*]] = arith.addi %arg4, %arg5 : index
// CHECK-DAG:   %[[T1:.*]] = arith.muli %arg4, %0 : index
// CHECK-DAG:   %[[T2:.*]] = arith.subi %1, %arg5 : index
// CHECK-DAG:   %[[PTR:.*]] = llvm.inttoptr %[[ARG0:.*]] : i64 to !llvm.ptr<1>
// CHECK-DAG:   %[[EXTRACT0:.*]] = tensor.extract %[[ARG3:.*]][%[[T0]], %[[T2]]] : tensor<64x64xi64>
// CHECK-DAG:   %[[EXTRACT1:.*]] = tensor.extract %[[ARG2:.*]][%[[T0]], %[[T2]]] : tensor<64x64xi64>
// CHECK:       %[[VAL1:.*]]:2 = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[ARG1:.*]], %[[ARG6:.*]] = %[[EXTRACT1]]) -> (tensor<64x64xf32>, i64) {
// CHECK:         %[[ADDI:.*]] = arith.addi %[[ARG6]], %[[EXTRACT0]] : i64
// CHECK:         %[[VAL2:.*]] = arith.index_cast %[[ARG6]] : i64 to index
// CHECK:         %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [64, %[[VAL2]]]
// CHECK:         %[[VAL3:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:         %[[VAL4:.*]] = tensor.extract_slice %[[VAL3]][%[[VAL2]], %[[VAL2]]] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
// CHECK:         %[[VAL_MAP:.*]] = linalg.map { math.absf } ins(%[[VAL4]] : tensor<64x64xf32>) outs(%[[ARG5]] : tensor<64x64xf32>)
// CHECK:         scf.yield %[[VAL_MAP]], %[[ADDI]] : tensor<64x64xf32>, i64
// CHECK:       }
// CHECK:       return %[[VAL1]]#0 : tensor<64x64xf32>

// -----
// COM: Fail due to use block argument as index.
func.func @extract_from_for_iter_args_fail_case0(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: scf.for %[[ARG0:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]] iter_args(%[[ARG1:.*]] = %[[ARG2:.*]], %[[ARG3:.*]] = %[[ARG4:.*]], %[[ARG5:.*]] = %[[C0:.*]]) -> (tensor<64x64xf32>, tensor<64x64xi64>, index)
  %0:3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2, %arg7 = %c0) -> (tensor<64x64xf32>, tensor<64x64xi64>, index) {
    %1 = tensor.extract %arg6[%c0, %arg7] : tensor<64x64xi64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [64, %2], strides: [1, 1] : !llvm.ptr<1> to memref<64x?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<64x?xf32, 1>
    %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %mapped = linalg.map { math.absf } ins(%4 : tensor<64x64xf32>) outs(%arg5 : tensor<64x64xf32>)
    %5 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%5 : tensor<64x64xi64>)
    %6 = arith.addi %arg7, %c1 : index
    scf.yield %mapped, %mapped_0, %6 : tensor<64x64xf32>, tensor<64x64xi64>, index
  }
  return %0#0 : tensor<64x64xf32>
}

// -----
// COM: Fail due to iter argument defines another yield operand.
func.func @extract_from_for_iter_args_fail_case1(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: scf.for %[[ARG0:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]] iter_args(%[[ARG1:.*]] = %[[ARG2:.*]], %[[ARG3:.*]] = %[[ARG4:.*]]) -> (tensor<64x64xf32>, tensor<64x64xi64>)
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
    %1 = tensor.extract %arg6[%c0, %c0] : tensor<64x64xi64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [64, %2], strides: [1, 1] : !llvm.ptr<1> to memref<64x?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<64x?xf32, 1>
    %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %5 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%5 : tensor<64x64xi64>)
    %6 = arith.sitofp %arg6 : tensor<64x64xi64> to tensor<64x64xf32>
    %mapped = linalg.map { arith.addf } ins(%4, %6 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg5 : tensor<64x64xf32>)
    scf.yield %mapped, %mapped_0 : tensor<64x64xf32>, tensor<64x64xi64>
  }
  return %0#0 : tensor<64x64xf32>
}

// -----
// COM: Fail due to the forward slice of iter argument depends on anther value inside loop.
func.func @extract_from_for_iter_args_fail_case2(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: scf.for %[[ARG0:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]] iter_args(%[[ARG1:.*]] = %[[ARG2:.*]], %[[ARG3:.*]] = %[[ARG4:.*]]) -> (tensor<64x64xf32>, tensor<64x64xi64>)
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
    %1 = tensor.extract %arg6[%c0, %c0] : tensor<64x64xi64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [64, %2], strides: [1, 1] : !llvm.ptr<1> to memref<64x?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<64x?xf32, 1>
    %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %mapped = linalg.map { math.absf } ins(%4 : tensor<64x64xf32>) outs(%arg5 : tensor<64x64xf32>)
    %5 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%5 : tensor<64x64xi64>)
    %6 = arith.fptosi %mapped : tensor<64x64xf32> to tensor<64x64xi64>
    %7 = tensor.empty() : tensor<64x64xi64>
    %mapped_1 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%mapped_0, %6 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%7 : tensor<64x64xi64>)
    scf.yield %mapped, %mapped_1 : tensor<64x64xf32>, tensor<64x64xi64>
  }
  return %0#0 : tensor<64x64xf32>
}

// -----
// CHECK-LABEL: func.func @extract_element_from_collapse_shape_op_with_single_element(
// CHECK-SAME:                                                                          %[[ARG0:.*]]: tensor<1xi32>) -> i32 {
// CHECK:           %[[INDEX:.*]] = arith.constant 0 : index
// CHECK:           %[[OUT:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[INDEX]]] : tensor<1xi32>
// CHECK:           return %[[OUT]] : i32
func.func @extract_element_from_collapse_shape_op_with_single_element(%arg0: tensor<1xi32>) -> i32 {
  %0 = tensor.collapse_shape %arg0 [] : tensor<1xi32> into tensor<i32>
  %1 = tensor.extract %0[] : tensor<i32>
  return %1 : i32
}

// -----
// COM: %0 %1 %2 %3 will be bufferized to the same memref. So, we can not move
// tensor.extract before %2.
// CHECK-LABEL:   func.func @test_destination_style_op_for_result_chain(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, f32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_1]]] : tensor<64xf32>
// CHECK:           %[[VAL_3:.*]] = math.absf %[[VAL_2]] : f32
// CHECK:           %[[VAL_4:.*]] = math.exp %[[VAL_3]] : f32
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_5]] : tensor<64xf32>)
// CHECK:           %[[VAL_7:.*]] = linalg.map { math.atan } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_6]] : tensor<64xf32>)
// CHECK:           return %[[VAL_7]], %[[VAL_4]] : tensor<64xf32>, f32
// CHECK:         }
func.func @test_destination_style_op_for_result_chain(%arg0: tensor<64xf32>) -> (tensor<64xf32>, f32) {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %4 = tensor.extract %3[%c0] : tensor<64xf32>
  return %2, %4 : tensor<64xf32>, f32
}

// -----
// CHECK-LABEL:   func.func @test_destination_style_op_cross_block(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, f32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_1]]] : tensor<64xf32>
// CHECK:           %[[VAL_3:.*]] = math.absf %[[VAL_2]] : f32
// CHECK:           %[[VAL_4:.*]] = math.exp %[[VAL_3]] : f32
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_5]] : tensor<64xf32>)
// CHECK:           %[[VAL_7:.*]] = linalg.map { math.atan } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_6]] : tensor<64xf32>)
// CHECK:           return %[[VAL_7]], %[[VAL_4]] : tensor<64xf32>, f32
// CHECK:         }
func.func @test_destination_style_op_cross_block(%arg0: tensor<64xf32>) -> (tensor<64xf32>, f32) {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  cf.br ^bb1
^bb1:
  %4 = tensor.extract %3[%c0] : tensor<64xf32>
  return %2, %4 : tensor<64xf32>, f32
}

// -----
// CHECK-LABEL:   func.func @test_destination_style_op_for_init_chain(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: tensor<64xf32>) -> (tensor<64xf32>, f32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_1]]] : tensor<64xf32>
// CHECK:           %[[VAL_3:.*]] = math.atan %[[VAL_2]] : f32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:           %[[VAL_5:.*]] = linalg.map { math.absf } ins(%[[VAL_0]] : tensor<64xf32>) outs(%[[VAL_4]] : tensor<64xf32>)
// CHECK:           %[[VAL_6:.*]] = linalg.map { math.exp } ins(%[[VAL_5]] : tensor<64xf32>) outs(%[[VAL_5]] : tensor<64xf32>)
// CHECK:           return %[[VAL_6]], %[[VAL_3]] : tensor<64xf32>, f32
// CHECK:         }
func.func @test_destination_style_op_for_init_chain(%arg0: tensor<64xf32>) -> (tensor<64xf32>, f32) {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.map { math.absf } ins(%arg0: tensor<64xf32>) outs(%0: tensor<64xf32>)
  %2 = linalg.map { math.atan } ins(%arg0: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %3 = linalg.map { math.exp } ins(%1: tensor<64xf32>) outs(%1: tensor<64xf32>)
  %4 = tensor.extract %2[%c0] : tensor<64xf32>
  return %3, %4 : tensor<64xf32>, f32
}

// COM: 0d test cases.
// -----
// CHECK-LABEL: @fill_0d
func.func @fill_0d(%arg0: i64) -> i64 {
  // CHECK: return %[[ARG0:.*]] : i64
  // CHECK-NOT: linalg.*
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<i64>
  %1 = linalg.fill ins(%arg0 : i64) outs(%0 : tensor<i64>) -> tensor<i64>
  %2 = tensor.extract %1[] : tensor<i64>
  return %2 : i64
}

// -----
// CHECK-LABEL: @arith_addi_0d
func.func @arith_addi_0d(%arg0: tensor<i32>, %arg1: tensor<i32>) -> i32 {
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %[[ARG1:.*]][] : tensor<i32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %[[ARG0:.*]][] : tensor<i32>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : i32
  // CHECK-NOT: linalg.*
  %0 = arith.addi %arg0, %arg1 : tensor<i32>
  %1 = tensor.extract %0[] : tensor<i32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @math_erf_0d
func.func @math_erf_0d(%arg0: tensor<f32>) -> f32 {
  // CHECK-NEXT: %[[ARG_EXTRA:.*]] = tensor.extract %[[ARG0:.*]][] : tensor<f32>
  // CHECK-NEXT: %[[ERF:.*]] = math.erf %[[ARG_EXTRA]] : f32
  // CHECK-NOT: linalg.*
  %0 = math.erf %arg0 : tensor<f32>
  %1 = tensor.extract %0[] : tensor<f32>
  return %1 : f32
}

// -----
// CHECK-LABEL: @map_0d
func.func @map_0d(%arg0: tensor<i32>, %arg1: tensor<i32>) -> i32 {
  // CHECK-NEXT: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[] : tensor<i32>
  // CHECK-NEXT: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[] : tensor<i32>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG_EXTRA1]], %[[ARG_EXTRA0]] : i32
  // CHECK-NOT:  linalg.*
  %0 = tensor.empty() : tensor<i32>
  %1 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<i32>, tensor<i32>) outs(%0 : tensor<i32>)
  %2 = tensor.extract %1[] : tensor<i32>
  return %2 : i32
}

// -----
// CHECK-LABEL: @map_0d_with_two_payloads
func.func @map_0d_with_two_payloads(%arg0: tensor<i64>, %arg1: tensor<i32>) -> i64 {
  // CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  %c1 = arith.constant 1 : i64
  // CHECK: %[[ARG_EXTRA0:.*]] = tensor.extract %arg1[] : tensor<i32>
  // CHECK: %[[ARG_EXTRA1:.*]] = tensor.extract %arg0[] : tensor<i64>
  // CHECK: %[[EXT:.*]] = arith.extsi %[[ARG_EXTRA0]] : i32 to i64
  // CHECK: %[[ADD_0:.*]] = arith.addi %[[ARG_EXTRA1]], %[[EXT]] : i64
  // CHECK: %[[ADD_1:.*]] = arith.addi %[[ADD_0]], %[[C1_I64]] : i64
  %0 = tensor.empty() : tensor<i64>
  %1 = linalg.map ins(%arg0, %arg1 : tensor<i64>, tensor<i32>) outs(%0 : tensor<i64>)
  (%in: i64, %in1:i32) {
    %2 = arith.extsi %in1 : i32 to i64
    %3 = arith.addi %in, %2 : i64
    %4 = arith.addi %3, %c1 : i64
    linalg.yield %4 : i64
  }
  %5 = tensor.extract %1[] : tensor<i64>
  return %5 : i64
}

// -----
// CHECK-LABEL: @broadcast_0d
func.func @broadcast_0d(%arg0: tensor<i32>) -> i32 {
  // CHECK-NEXT: %[[EXTRACT:.*]] = tensor.extract %arg0[] : tensor<i32>
  // CHECK-NOT:  linalg.broadcast
  %c5 = arith.constant 5 : index
  %1 = tensor.empty() : tensor<128xi32>
  %2 = linalg.broadcast ins(%arg0: tensor<i32>) outs(%1: tensor<128xi32>) dimensions = [0]
  %3 = tensor.extract %2[%c5] : tensor<128xi32>
  return %3 : i32
}

// -----
// CHECK-LABEL: @collapse_shape_op_0d
func.func @collapse_shape_op_0d(%arg0: tensor<1x1xi32>) -> i32 {
  // CHECK-DAG: %[[C0_INDEX:.*]] = arith.constant 0 : index
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %[[ARG0:.*]][%[[C0_INDEX]], %[[C0_INDEX]]] : tensor<1x1xi32>
  // CHECK-NOT: tensor.collapse_shape
  %0 = tensor.collapse_shape %arg0 [] : tensor<1x1xi32> into tensor<i32>
  %1 = tensor.extract %0[] : tensor<i32>
  return %1 : i32
}

// -----
// CHECK-LABEL: @expand_shape_op_0d
func.func @expand_shape_op_0d(%arg0: tensor<i32>) -> i32 {
  // CHECK:     %[[EXTRACT:.*]] = tensor.extract %[[ARG0:.*]][] : tensor<i32>
  // CHECK-NOT: tensor.collapse_shape
  %c0 = arith.constant 0 : index
  %0 = tensor.expand_shape %arg0 [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
  %1 = tensor.extract %0[%c0, %c0] : tensor<1x1xi32>
  return %1 : i32
}

// -----
func.func @for_op_with_0d_iter_args(%arg0: i64, %arg1: tensor<f32>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<f32>, tensor<i64>) {
    %1 = tensor.extract %arg6[] : tensor<i64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [%2], strides: [1] : !llvm.ptr<1> to memref<?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<?xf32, 1>
    %4 = tensor.extract_slice %3[%2] [1] [1] : tensor<?xf32> to tensor<f32>
    %mapped = linalg.map { math.absf } ins(%4 : tensor<f32>) outs(%arg5 : tensor<f32>)
    %5 = tensor.empty() : tensor<i64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<i64>, tensor<i64>) outs(%5 : tensor<i64>)
    scf.yield %mapped, %mapped_0 : tensor<f32>, tensor<i64>
  }
  return %0#0 : tensor<f32>
}
// CHECK-LABEL: @for_op_with_0d_iter_args
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[PTR:.*]] = llvm.inttoptr %[[ARG0:.*]] : i64 to !llvm.ptr<1>
// CHECK-DAG:   %[[EXTRACT0:.*]] = tensor.extract %[[ARG3:.*]][] : tensor<i64>
// CHECK-DAG:   %[[EXTRACT1:.*]] = tensor.extract %[[ARG2:.*]][] : tensor<i64>
// CHECK:       %[[VAL1:.*]]:2 = scf.for %[[ARG4:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG5:.*]] = %[[ARG1:.*]], %[[ARG6:.*]] = %[[EXTRACT1]]) -> (tensor<f32>, i64) {
// CHECK:         %[[ADD:.*]] = arith.addi %[[ARG6]], %[[EXTRACT0]] : i64
// CHECK:         %[[VAL2:.*]] = arith.index_cast %[[ARG6]] : i64 to index
// CHECK:         %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [%[[VAL2]]]
// CHECK:         %[[VAL3:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:         %[[VAL4:.*]] = tensor.extract_slice %[[VAL3]][%[[VAL2]]] [1] [1] : tensor<?xf32> to tensor<f32>
// CHECK:         %[[VAL_MAP:.*]] = linalg.map { math.absf } ins(%[[VAL4]] : tensor<f32>) outs(%[[ARG5]] : tensor<f32>)
// CHECK:         scf.yield %[[VAL_MAP]], %[[ADD]] : tensor<f32>, i64
// CHECK:       }

// -----
func.func @extract_from_for_iter_args_failed(%arg0: i64, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xi64>, %arg3: tensor<64x64xi64>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
    %1 = tensor.extract %arg6[%c0, %c0] : tensor<64x64xi64>
    %2 = arith.index_cast %1 : i64 to index
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
    %memref = aux.view %ptr to offset: [0], sizes: [64, %2], strides: [1, 1] : !llvm.ptr<1> to memref<64x?xf32, 1>
    %3 = bufferization.to_tensor %memref : memref<64x?xf32, 1>
    %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
    %mapped = linalg.map { math.absf } ins(%4 : tensor<64x64xf32>) outs(%arg5 : tensor<64x64xf32>)
    %5 = tensor.empty() : tensor<64x64xi64>
    %mapped_0 = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>) outs(%5 : tensor<64x64xi64>)
    %6 = tensor.empty() : tensor<64x64xi64>
    %transpose = linalg.transpose ins(%mapped_0 : tensor<64x64xi64>) outs(%6 : tensor<64x64xi64>) permutation = [1, 0]
    scf.yield %mapped, %transpose : tensor<64x64xf32>, tensor<64x64xi64>
  }
  return %0#0 : tensor<64x64xf32>
}

// CHECK-LABEL: @extract_from_for_iter_args_failed
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<64x64xf32>, %[[ARG2:.*]]: tensor<64x64xi64>, %[[ARG3:.*]]: tensor<64x64xi64>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK:     %[[VAL_0:.*]]:2 = scf.for %[[ARG4:.*]] = %c0 to %c2 step %c1 iter_args(%[[ARG5:.*]] = %[[ARG1]], %[[ARG6:.*]] = %[[ARG2]]) -> (tensor<64x64xf32>, tensor<64x64xi64>) {
// CHECK:       %[[VAL_1:.*]] = tensor.extract %[[ARG6]][%c0, %c0] : tensor<64x64xi64>
// CHECK:       %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:       %[[VAL_3:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr<1>
// CHECK:       %[[VAL_4:.*]] = aux.view %[[VAL_3]] to offset: [0], sizes: [64, %[[VAL_2]]], strides: [1, 1] : <1> to memref<64x?xf32, 1>
// CHECK:       %[[VAL_5:.*]] = bufferization.to_tensor %[[VAL_4]] : memref<64x?xf32, 1>
// CHECK:       %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_5]][%[[VAL_2]], %[[VAL_2]]] [64, 64] [1, 1] : tensor<64x?xf32> to tensor<64x64xf32>
// CHECK:       %[[VAL_7:.*]] = linalg.map { math.absf } ins(%[[VAL_6]] : tensor<64x64xf32>) outs(%[[ARG5]] : tensor<64x64xf32>)
// CHECK:       %[[VAL_8:.*]] = tensor.empty() : tensor<64x64xi64>
// CHECK:       %[[VAL_9:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[ARG6]], %[[ARG3]] : tensor<64x64xi64>, tensor<64x64xi64>) outs(%[[VAL_8]] : tensor<64x64xi64>)
// CHECK:       %[[VAL_10:.*]] = tensor.empty() : tensor<64x64xi64>
// CHECK:       %[[VAL_11:.*]] = linalg.transpose ins(%[[VAL_9]] : tensor<64x64xi64>) outs(%[[VAL_10]] : tensor<64x64xi64>) permutation = [1, 0]
// CHECK:       scf.yield %[[VAL_7]], %[[VAL_11]] : tensor<64x64xf32>, tensor<64x64xi64>
// CHECK:     }
// CHECK:     return %[[VAL_0]]#0 : tensor<64x64xf32>
