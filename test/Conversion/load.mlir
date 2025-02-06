// RUN: triton-linalg-opt --convert-triton-to-linalg %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @load
// CHECK-SAME: %[[ARG:.*]]: i64
tt.func @load(%arg0: !tt.ptr<f32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%c0]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [128], strides: [1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<128xf32>) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>
  %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_stride_unknown
// CHECK-SAME: %[[ARG:.*]]: i64
tt.func @load_stride_unknown(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = arith.muli %1, %2: tensor<128xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  %4 = tt.addptr %0, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%c0]
  // CHECK: %[[OFFSET1:.*]] = tensor.extract %[[ADD]][%c1]
  // CHECK-NEXT: %[[STRIDE:.*]] = arith.subi %[[OFFSET1]], %[[OFFSET]]
  // CHECK-NEXT: %[[STRIDE_INDEX:.*]] = arith.index_cast %[[STRIDE]]
  // CHECK-NEXT: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [128], strides: [%[[STRIDE_INDEX]]]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<128xf32>) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>
  %5 = tt.load %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_masked
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32
tt.func @load_masked(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: linalg_ext.pad
  // CHECK: %[[ARG1_INDEX:.*]] = arith.index_cast %[[ARG1]]
  // CHECK: %[[ARG1_INDEX1:.*]] = arith.maxsi %[[ARG1_INDEX]], %[[C0:.*]]
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[C128:.*]], %[[ARG1_INDEX1]] : index
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  %ldptr = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %others = arith.constant dense<1.0> : tensor<128xf32>
  %2 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %mask = arith.cmpi slt, %1, %2 : tensor<128xi32>
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%[[C0:.*]]]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [%[[SIZE]]], strides: [1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK: %[[EMPTY:.*]] = tensor.empty
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<?xf32>) outs(%[[EMPTY]] : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: linalg_ext.pad
  %buff = tt.load %ldptr, %mask, %others {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_masked_with_ub_lb
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32
// CHECK: linalg_ext.pad
// CHECK: %[[ARG1_INDEX:.*]] = arith.index_cast %[[ARG1]]
// CHECK: %[[ARG1_INDEX1:.*]] = arith.maxsi %[[ARG1_INDEX]], %{{.*}}
// CHECK: %[[ARG1_INDEX2:.*]] = arith.minsi %{{.*}}, %[[ARG1_INDEX1]]
// CHECK: %[[ARG2_INDEX:.*]] = arith.index_cast %[[ARG2]]
// CHECK: %[[ARG2_INDEX1:.*]] = arith.addi %[[ARG2_INDEX:.*]], %[[C1:.*]] : index
// CHECK: %[[ARG2_INDEX2:.*]] = arith.minsi %{{.*}}, %[[ARG2_INDEX1:.*]] : index
// CHECK: %[[ARG2_INDEX3:.*]] = arith.maxsi %{{.*}}, %[[ARG2_INDEX2:.*]] : index
// CHECK: %[[LB:.*]] = arith.maxsi %{{.*}}, %[[ARG2_INDEX3]] : index
// CHECK: %[[UB:.*]] = arith.minsi %[[ARG1_INDEX2:.*]], %{{.*}} : index
// CHECK: %[[LEN:.*]] = arith.subi %[[UB]], %[[LB]] : index
// CHECK: %[[SIZE:.*]] = arith.maxsi %[[LEN]], %{{.*}} : index
// CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
// CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%[[LB]]]
// CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
// CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [%[[SIZE]]], strides: [1]
// CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<?xf32>) outs(%[[EMPTY]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK: linalg_ext.pad
tt.func @load_masked_with_ub_lb(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ldptr = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %others = arith.constant dense<1.0> : tensor<128xf32>
  %2 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %lb = arith.cmpi slt, %1, %2 : tensor<128xi32>
  %3 = tt.splat %arg2 : i32 -> tensor<128xi32>
  %ub = arith.cmpi sgt, %1, %3 : tensor<128xi32>
  %mask = arith.andi %lb, %ub : tensor<128xi1>
  %buff = tt.load %ldptr, %mask, %others {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_broadcast
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32
tt.func @load_broadcast(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %2 = tt.splat %arg1 : i32 -> tensor<8xi32>
  %3 = arith.cmpi slt, %1, %2 : tensor<8xi32>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<8xi1> -> tensor<1x8xi1>
  %mask = tt.broadcast %4 : tensor<1x8xi1> -> tensor<4x8xi1>
  %5 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<8x!tt.ptr<f32>> -> tensor<1x8x!tt.ptr<f32>>
  %7 = tt.broadcast %6 : tensor<1x8x!tt.ptr<f32>> -> tensor<4x8x!tt.ptr<f32>>
  // CHECK: %[[BROADCAST:.*]] = linalg.broadcast{{.*}}4x8xi32
  // CHECK: tensor.extract
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_2:.*]] = arith.constant 0 : index
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[BROADCAST]][%[[C0_1]], %[[C0_2]]]
  // CHECK-NEXT: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [%[[SIZE:.*]]], strides: [1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK: %[[EMPTY:.*]] = tensor.empty
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<?xf32>) outs(%[[EMPTY]] : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: linalg.broadcast
  %buff = tt.load %7, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4x8x!tt.ptr<f32>>
  tt.return
}


// -----
// CHECK-LABEL: @load_2d
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64
tt.func @load_2d(%arg0: i32, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %offset = tt.splat %arg0 : i32 -> tensor<64xi32>
  %stride = tt.splat %arg1 : i32 -> tensor<64x1xi32>
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = arith.addi %offset, %2 : tensor<64xi32>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %6 = arith.muli %5, %stride : tensor<64x1xi32>
  %7 = tt.broadcast %6 : tensor<64x1xi32> -> tensor<64x64xi32>
  %8 = tt.broadcast %4 : tensor<1x64xi32> -> tensor<64x64xi32>
  %9 = arith.addi %7, %8 : tensor<64x64xi32>
  %10 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
  %11 = tt.addptr %10, %9 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>

  // CHECK: %[[RHS:.*]] = linalg.fill ins(%[[C0:.*]]: i32) outs(%[[FILL_INIT:.*]] : tensor<64x64xi32>) -> tensor<64x64xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[LHS:.*]], %[[RHS]] : tensor<64x64xi32>, tensor<64x64xi32>) outs(%[[INITS:.*]] : tensor<64x64xi32>)
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%c0, %c0]
  // CHECK: %[[OFFSET10:.*]] = tensor.extract %[[ADD]][%c1, %c0]
  // CHECK-NEXT: %[[STRIDE1:.*]] = arith.subi %[[OFFSET10]], %[[OFFSET]]
  // CHECK-NEXT: %[[STRIDE1_INDEX:.*]] = arith.index_cast %[[STRIDE1]]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [64, 64], strides: [%[[STRIDE1_INDEX]], 1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<64x64xf32>) outs(%[[EMPTY]] : tensor<64x64xf32>) -> tensor<64x64xf32>
  %data = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK: #map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-LABEL: @load_3d_with_dim_1
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64
tt.func @load_3d_with_dim_1(%arg0: i32, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %offset = tt.splat %arg0 : i32 -> tensor<64xi32>
  %stride = tt.splat %arg1 : i32 -> tensor<64x1xi32>
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = arith.addi %offset, %2 : tensor<64xi32>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %6 = arith.muli %5, %stride : tensor<64x1xi32>
  %7 = tt.broadcast %6 : tensor<64x1xi32> -> tensor<64x64xi32>
  %8 = tt.broadcast %4 : tensor<1x64xi32> -> tensor<64x64xi32>
  %9 = arith.addi %7, %8 : tensor<64x64xi32>
  %10 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
  %11 = tt.addptr %10, %9 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
  %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64x64x!tt.ptr<f32>> -> tensor<64x1x64x!tt.ptr<f32>>

  // CHECK: %[[RHS:.*]] = linalg.fill ins(%[[C0:.*]]: i32) outs(%[[FILL_INIT:.*]] : tensor<64x64xi32>) -> tensor<64x64xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[LHS:.*]], %[[RHS]] : tensor<64x64xi32>, tensor<64x64xi32>) outs(%[[INITS:.*]] : tensor<64x64xi32>)
  // CHECK: %[[EXPAND_ADD:.*]] = tensor.expand_shape %[[ADD]]
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[EXPAND_ADD]][%c0, %c0, %c0]
  // CHECK: %[[OFFSET100:.*]] = tensor.extract %[[EXPAND_ADD]][%c1, %c0, %c0]
  // CHECK-NEXT: %[[STRIDE0:.*]] = arith.subi %[[OFFSET100]], %[[OFFSET]]
  // CHECK-NEXT: %[[STRIDE0_INDEX:.*]] = arith.index_cast %[[STRIDE0]]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[VIEW_MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [64, 64], strides: [%[[STRIDE0_INDEX]], 1] : !llvm.ptr to memref<64x64xf32, #map>
  // CHECK-NEXT: %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[VIEW_MEMREF]] restrict writable : memref<64x64xf32, #map>
  // CHECK: %[[COPY:.*]] = linalg.copy ins(%[[TO_TENSOR]] : tensor<64x64xf32>)
  // CHECK: %[[BROADCAST:.*]] = linalg.broadcast ins(%[[COPY]] : tensor<64x64xf32>) {{.*}} dimensions = [1]
  %data = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x1x64x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_transpose
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i64
tt.func @load_transpose_2d(%arg0: i32, %arg1: i32, %arg2: !tt.ptr<f32>) {
  %offset = tt.splat %arg0 : i32 -> tensor<64xi32>
  %stride = tt.splat %arg1 : i32 -> tensor<1x64xi32>
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = arith.addi %offset, %2 : tensor<64xi32>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %5 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %6 = arith.muli %5, %stride : tensor<1x64xi32>
  %7 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<64x64xi32>
  %8 = tt.broadcast %4 : tensor<64x1xi32> -> tensor<64x64xi32>
  %9 = arith.addi %7, %8 : tensor<64x64xi32>
  %10 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
  %11 = tt.addptr %10, %9 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>

  // CHECK: %[[RHS:.*]] = linalg.fill ins(%[[C0:.*]]: i32) outs(%[[FILL_INIT:.*]] : tensor<64x64xi32>) -> tensor<64x64xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%[[LHS:.*]], %[[RHS]] : tensor<64x64xi32>, tensor<64x64xi32>) outs(%[[INITS:.*]] : tensor<64x64xi32>)
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%c0, %c0]
  // CHECK: %[[OFFSET01:.*]] = tensor.extract %[[ADD]][%c0, %c1]
  // CHECK-NEXT: %[[STRIDE1:.*]] = arith.subi %[[OFFSET01]], %[[OFFSET]]
  // CHECK-NEXT: %[[STRIDE1_INDEX:.*]] = arith.index_cast %[[STRIDE1]]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [64, 64], strides: [%[[STRIDE1_INDEX]], 1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<64x64xf32>) outs(%[[EMPTY]] : tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK: linalg.transpose
  %data = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_scalar
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32
tt.func @load_scalar(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[PTR2:.*]] = llvm.getelementptr %[[PTR]][%[[ARG1]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR2]] : !llvm.ptr to i64
  // CHECK-NEXT: %[[PTR3:.*]] = llvm.inttoptr %[[INT]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR3]] to offset: [0], sizes: [1], strides: [1] : !llvm.ptr to memref<1xf32>
  // CHECK: memref.load %[[MEMREF]][%c0] : memref<1xf32>
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
  tt.return
}

// -----
// CHECK-LABEL: @load_scalar_masked
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i1
tt.func @load_scalar_masked(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i1) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[PTR2:.*]] = llvm.getelementptr %[[PTR]][%[[ARG1]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR2]] : !llvm.ptr to i64
  // CHECK-NEXT: %[[PTR3:.*]] = llvm.inttoptr %[[INT]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR3]] to offset: [0], sizes: [1], strides: [1]
  // CHECK: %[[DATA:.*]] = memref.load %[[MEMREF]][%c0]
  // CHECK-NEXT: scf.if %[[ARG2]] -> (f32) {
  // CHECK-NEXT: scf.yield %[[DATA]]
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.yield %[[DATA]]
  %1 = tt.load %0, %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
  tt.return
}

// -----
// CHECK-LABEL: @load_scalar_masked_other
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: f32
tt.func @load_scalar_masked_other(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i1, %arg3: f32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[PTR2:.*]] = llvm.getelementptr %[[PTR]][%[[ARG1]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR2]] : !llvm.ptr to i64
  // CHECK-NEXT: %[[PTR3:.*]] = llvm.inttoptr %[[INT]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR3]] to offset: [0], sizes: [1], strides: [1]
  // CHECK: %[[DATA:.*]] = memref.load %[[MEMREF]][%c0]
  // CHECK-NEXT: scf.if %[[ARG2]] -> (f32) {
  // CHECK-NEXT: scf.yield %[[DATA]]
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.yield %[[ARG3]]
  %1 = tt.load %0, %arg2, %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
  tt.return
}

// -----
// CHECK-LABEL: @load_from_constant
// CHECK-SAME: %[[ARG0:.*]]: i64
tt.func @load_from_constant(%arg0: !tt.ptr<f32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[C0:.*]]: i32)
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[FILL]][%c0]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [], strides: []
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: %[[NEW_TENSOR:.*]] = linalg.copy ins(%[[TENSOR]] : tensor<f32>) outs(%[[EMPTY]] : tensor<f32>) -> tensor<f32>
  // CHECK: linalg.broadcast ins(%[[NEW_TENSOR]]
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_from_constant_masked
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i32
tt.func @load_from_constant_masked(%arg0: !tt.ptr<f32>, %arg1: i32) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %2 = tt.splat %arg1 : i32 -> tensor<128xi32>
  %mask = arith.cmpi slt, %1, %2 : tensor<128xi32>
  %others = arith.constant dense<1.0> : tensor<128xf32>
  // CHECK: %[[MASK:.*]] = linalg_ext.pad ins({{.*}} : tensor<?xi1>)
  // CHECK: %[[ZEROS:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : tensor<128xi32>) -> tensor<128xi32>
  // CHECK-DAG: %[[INDICES:.*]] = tensor.expand_shape %[[ZEROS]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
  // CHECK-DAG: %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [9223372036854775807], strides: [1]
  // CHECK-DAG: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK: linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%[[TENSOR]], %[[INDICES]], %[[MASK]] : tensor<9223372036854775807xf32>, tensor<128x1xi32>, tensor<128xi1>)
  %buff = tt.load %0, %mask, %others {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_bitcast
// CHECK-SAME: %[[ARG0:.*]]: i64
tt.func @load_bitcast(%arg0: !tt.ptr<i1>) {
  %0 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
  %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
  // CHECK: %[[OFFSET:.*]] = tensor.extract %[[ADD]][%c0]
  // CHECK: %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]]
  // CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-NEXT: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [128], strides: [1]
  // CHECK-NEXT: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]
  // CHECK-NEXT: %[[EMPTY:.*]] = tensor.empty()
  // CHECK-NEXT: linalg.copy ins(%[[TENSOR]] : tensor<128xi8>) outs(%[[EMPTY]] : tensor<128xi8>) -> tensor<128xi8>
  %4 = tt.load %3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<i8>>
  tt.return
}

// -----
tt.func @load_invalid_origin_ptr(%arg0: tensor<128x!tt.ptr<f32>>) {
  // expected-error @+1 {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

///////////////////////////////// load gather //////////////////////////////

// -----
// CHECK-LABEL: @load_gather_1d
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128xi32>
tt.func @load_gather_1d(%arg0: !tt.ptr<f32>, %arg1: tensor<128xi32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>

  // CHECK-DAG: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK-DAG: %[[INDICES:.*]] = tensor.expand_shape %[[ADD]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>

  // CHECK-DAG: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [9223372036854775807], strides: [1]
  // CHECK-DAG: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]

  // CHECK: linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%[[TENSOR]], %[[INDICES]]

  %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_gather_2d
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128x64xi32>
tt.func @load_gather_2d(%arg0: !tt.ptr<f32>, %arg1: tensor<128x64xi32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>

  // CHECK-DAG: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK-DAG: %[[INDIECS_COLLAPSE:.*]] = tensor.collapse_shape %[[ADD]] {{\[\[}}0, 1]] : tensor<128x64xi32> into tensor<8192xi32>
  // CHECK-DAG: %[[INDICES:.*]] = tensor.expand_shape %[[INDIECS_COLLAPSE]] {{\[\[}}0, 1]] output_shape [8192, 1] : tensor<8192xi32> into tensor<8192x1xi32>

  // CHECK-DAG: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [9223372036854775807], strides: [1]
  // CHECK-DAG: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]

  // CHECK: linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%[[TENSOR]], %[[INDICES]]
  %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_gather_0d_mask
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<i32>, %[[ARG2:.*]]: tensor<i1>
tt.func @load_gather_0d_mask(%arg0: !tt.ptr<f32>, %arg1: tensor<i32>, %arg2: tensor<i1>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<!tt.ptr<f32>>, tensor<i32>

  // CHECK-DAG: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK-DAG: %[[INDICES:.*]] = tensor.expand_shape %[[ADD]] [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>

  // CHECK-DAG: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [9223372036854775807], strides: [1]
  // CHECK-DAG: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]

  // CHECK-DAG: %[[MASK:.*]] = tensor.expand_shape %[[ARG2]] [] output_shape [1] : tensor<i1> into tensor<1xi1>

  // CHECK: linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%[[TENSOR]], %[[INDICES]], %[[MASK]]
  %2 = tt.load %1, %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL: @load_gather_1d_mask_other
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128xi32>, %[[ARG2:.*]]: tensor<128xi1>, %[[ARG3:.*]]: tensor<128xf32>
tt.func @load_gather_1d_mask_other(%arg0: !tt.ptr<f32>, %arg1: tensor<128xi32>, %arg2: tensor<128xi1>, %arg3: tensor<128xf32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.addptr %0, %arg1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>

  // CHECK-DAG: %[[ADD:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK-DAG: %[[INDICES:.*]] = tensor.expand_shape %[[ADD]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>

  // CHECK-DAG: %[[PTR:.*]] = llvm.inttoptr %[[ARG]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: [0], sizes: [9223372036854775807], strides: [1]
  // CHECK-DAG: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]]

  // CHECK: %[[GATHER_RES:.*]] = linalg_ext.gather dimension_map = [0] ranged_data(false) signed_indice(true) ins(%[[TENSOR]], %[[INDICES]], %[[ARG2]] : tensor<9223372036854775807xf32>, tensor<128x1xi32>, tensor<128xi1>)
  // CHECK: %[[COLLAPSED_RES:.*]] = tensor.collapse_shape %[[GATHER_RES:.*]] {{\[\[}}0, 1]] : tensor<128x1xf32> into tensor<128xf32>
  %2 = tt.load %1, %arg2, %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
module attributes {triton.is_linear= true} {
// CHECK-LABEL: @load_noncont_mask_with_other_true_linear
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128xi1>
tt.func @load_noncont_mask_with_other_true_linear(%arg0: !tt.ptr<f32>, %arg1: tensor<128xi1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ldptr = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %others = arith.constant dense<1.0> : tensor<128xf32>
  // CHECK:  %[[ADDI:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK:  %[[OFFSET:.*]] = tensor.extract %[[ADDI]][%[[C0]]] : tensor<128xi32>
  // CHECK:  %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]] : i32 to index
  // CHECK:  %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK:  %[[VIEW:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [128], strides: [1] : !llvm.ptr to memref<128xf32, #map>
  // CHECK:  %[[BUFF:.*]] = bufferization.to_tensor %[[VIEW]] restrict writable : memref<128xf32, #map>
  // CHECK:  %[[EMPTY:.*]] = tensor.empty()
  // CHECK:  %[[TENSOR:.*]] = linalg.copy ins(%[[BUFF]] : tensor<128xf32>) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>
  // CHECK:  linalg_ext.pad ins(%[[TENSOR]] : tensor<128xf32>)
  // CHECK:  ^bb0(%arg2: f32):
  // CHECK:    linalg_ext.yield %arg2 : f32
  // CHECK:  } -> tensor<128xf32>
  // CHECK:  linalg.map { arith.select }
  %buff = tt.load %ldptr, %arg1, %others {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return %buff : tensor<128xf32>
}
}

// -----
module attributes {triton.is_linear= false} {
tt.func @load_noncont_mask_with_other_false_linear(%arg0: !tt.ptr<f32>, %arg1: tensor<128xi1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ldptr = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %others = arith.constant dense<1.0> : tensor<128xf32>
  // CHECK: linalg_ext.gather
  %buff = tt.load %ldptr, %arg1, %others {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return %buff : tensor<128xf32>
}
}

// -----
module attributes {triton.is_linear= true} {
// CHECK-LABEL: @load_noncont_mask_without_other
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: tensor<128xi1>
tt.func @load_noncont_mask_without_other(%arg0: !tt.ptr<f32>, %arg1: tensor<128xi1>) -> tensor<128xf32> {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %ldptr = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK:  %[[ADDI:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} }
  // CHECK:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK:  %[[OFFSET:.*]] = tensor.extract %[[ADDI]][%[[C0]]] : tensor<128xi32>
  // CHECK:  %[[OFFSET_INDEX:.*]] = arith.index_cast %[[OFFSET]] : i32 to index
  // CHECK:  %[[PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
  // CHECK:  %[[VIEW:.*]] = aux.view %[[PTR]] to offset: [%[[OFFSET_INDEX]]], sizes: [128], strides: [1] : !llvm.ptr to memref<128xf32, #map>
  // CHECK:  %[[BUFF:.*]] = bufferization.to_tensor %[[VIEW]] restrict writable : memref<128xf32, #map>
  // CHECK:  %[[EMPTY:.*]] = tensor.empty()
  // CHECK:  %[[TENSOR:.*]] = linalg.copy ins(%[[BUFF]] : tensor<128xf32>) outs(%[[EMPTY]] : tensor<128xf32>) -> tensor<128xf32>
  // CHECK:  linalg_ext.pad ins(%[[TENSOR]] : tensor<128xf32>)
  // CHECK:  ^bb0(%arg2: f32):
  // CHECK:    linalg_ext.yield %arg2 : f32
  // CHECK:  } -> tensor<128xf32>
  %buff = tt.load %ldptr, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return %buff : tensor<128xf32>
}
}

// -----
tt.func @load_cache_ca(%arg0: !tt.ptr<f32>) {
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %2 = tt.addptr %0, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK: aux.view{{.*}}{cache_mode = "cmnormal"}
  %3 = tt.load %2 {cache = 2 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----
// CHECK-LABEL:   func.func @load__axis_info_0d_64bitdata
tt.func @load__axis_info_0d_64bitdata(%arg0: !tt.ptr<f64>) -> tensor<f64> {
  %0 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<!tt.ptr<f64>>
  // CHECK: %[[INDICE:.*]] = tensor.extract %[[INDICE_TENSOR:.*]][] : tensor<i32>
  // CHECK: %[[OFFSET:.*]] = arith.index_cast %[[INDICE]] : i32 to index
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[INT_PTR:.*]] : i64 to !llvm.ptr
  // CHECK: %[[MEMREF:.*]] = aux.view %[[PTR]] to offset: {{\[}}%[[OFFSET]]], sizes: [], strides: [] : !llvm.ptr to memref<f64, #map>
  // CHECK: %[[RES:.*]] = bufferization.to_tensor %[[MEMREF]] restrict writable : memref<f64, #map>
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<!tt.ptr<f64>>
  tt.return %1 : tensor<f64>
}
