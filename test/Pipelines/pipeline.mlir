// RUN: triton-linalg-opt %s -triton-to-linalg -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @add_kernel_01234
// CHECK-SAME: (%[[ARG0:.+]]: i64, %[[ARG1:.+]]: i64, %[[ARG2:.+]]: i64, %[[ARG3:.+]]: i32)
// CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG: %[[C1024_I32:.+]] = arith.constant 1024 : i32
// CHECK-DAG: %[[C0_F32:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL0:.+]] = tt.get_program_id x : i32
// CHECK: %[[VAL1:.+]] = arith.muli %[[VAL0]], %[[C1024_I32]] : i32
// CHECK: %[[VAL2:.+]] = arith.index_cast %[[VAL1]] : i32 to index
// CHECK: %[[VAL3:.+]] = arith.addi %[[VAL2]], %[[C1024]] : index
// CHECK: %[[VAL4:.+]] = arith.index_cast %[[ARG3]] : i32 to index
// CHECK: %[[VAL5:.+]] = arith.maxsi %[[VAL4]], %[[VAL2]] : index
// CHECK: %[[VAL6:.+]] = arith.minsi %[[VAL3]], %[[VAL5]] : index
// CHECK: %[[VAL7:.+]] = arith.subi %[[VAL6]], %[[VAL2]] : index
// CHECK: %[[VAL8:.+]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr
// CHECK: %[[VIEW:.+]] = aux.view %[[VAL8]] to offset: [%[[VAL2]]], sizes: [%[[VAL7]]], strides: [1] : !llvm.ptr to memref<?xf32, #map>
// CHECK: %[[VAL9:.+]] = bufferization.to_tensor %[[VIEW]] restrict writable : memref<?xf32, #map>
// CHECK: %[[VAL10:.+]] = tensor.empty(%[[VAL7]]) : tensor<?xf32>
// CHECK: %[[VAL11:.+]] = linalg.copy ins(%[[VAL9]] : tensor<?xf32>) outs(%[[VAL10]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK: %[[VAL12:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[VAL13:.+]] = arith.subi %[[C1024]], %[[VAL7]] : index
// CHECK: %[[VAL14:.+]] = linalg_ext.pad ins(%[[VAL11]] : tensor<?xf32>) outs(%[[VAL12]] : tensor<1024xf32>) pvalue(%[[C0_F32]] : f32) low = [0] high = [%[[VAL13]]] {
// CHECK: ^bb0(%[[ARG4:.+]]: f32):
// CHECK:   linalg_ext.yield %[[ARG4]] : f32
// CHECK: } -> tensor<1024xf32>
// CHECK: %[[VAL15:.+]] = llvm.inttoptr %[[ARG1]] : i64 to !llvm.ptr
// CHECK: %[[VIEW0:.+]] = aux.view %[[VAL15]] to offset: [%[[VAL2]]], sizes: [%[[VAL7]]], strides: [1] : !llvm.ptr to memref<?xf32, #map>
// CHECK: %[[VAL16:.+]] = bufferization.to_tensor %[[VIEW0]] restrict writable : memref<?xf32, #map>
// CHECK: %[[VAL17:.+]] = linalg.copy ins(%[[VAL16]] : tensor<?xf32>) outs(%[[VAL10]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK: %[[VAL18:.+]] = linalg_ext.pad ins(%[[VAL17]] : tensor<?xf32>) outs(%[[VAL12]] : tensor<1024xf32>) pvalue(%[[C0_F32]] : f32) low = [0] high = [%[[VAL13]]] {
// CHECK: ^bb0(%[[ARG4]]: f32):
// CHECK:   linalg_ext.yield %[[ARG4]] : f32
// CHECK: } -> tensor<1024xf32>
// CHECK: %[[MAPPED:.+]] = linalg.map { arith.addf } ins(%[[VAL14]], %[[VAL18]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[VAL12]] : tensor<1024xf32>)
// CHECK: %[[VAL19:.+]] = llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr
// CHECK: %[[VIEW1:.+]] = aux.view %[[VAL19]] to offset: [%[[VAL2]]], sizes: [%[[VAL7]]], strides: [1] : !llvm.ptr to memref<?xf32, #map>
// CHECK: %[[EXTRACED:.+]] = tensor.extract_slice %[[MAPPED]][0] [%[[VAL7]]] [1] : tensor<1024xf32> to tensor<?xf32>
// CHECK: bufferization.materialize_in_destination %[[EXTRACED]] in writable %[[VIEW1]] : (tensor<?xf32>, memref<?xf32, #map>) -> ()
// CHECK: return

tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
  %c1024_i32 = arith.constant 1024 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c1024_i32 : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %3 = tt.splat %1 : i32 -> tensor<1024xi32>
  %4 = arith.addi %3, %2 : tensor<1024xi32>
  %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
  %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
  %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x!tt.ptr<f32>>
  %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x!tt.ptr<f32>>
  %13 = arith.addf %9, %12 : tensor<1024xf32>
  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<1024x!tt.ptr<f32>>
  tt.return
}
