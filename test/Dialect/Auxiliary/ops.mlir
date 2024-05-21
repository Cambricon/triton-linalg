// RUN: triton-linalg-opt %s -split-input-file --allow-unregistered-dialect | FileCheck %s
// -----
// CHECK-LABEL: func @view
func.func @view(%in: !llvm.ptr)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %out = aux.view %in to
           offset: [%c0], sizes: [10, %c10], strides: [%c10, 1]
           :  !llvm.ptr to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// -----
// CHECK-LABEL: func @view_static_to_dynamic_sizes
func.func @view_static_to_dynamic_sizes(%in: !llvm.ptr)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %out = aux.view %in to
           offset: [1], sizes: [10, 10], strides: [1, 1]
           : !llvm.ptr to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// -----
// CHECK-LABEL: func @view_dynamic_offset
func.func @view_dynamic_offset(%in:  !llvm.ptr, %offset: index)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %out = aux.view %in to
           offset: [%offset], sizes: [10, 10], strides: [1, 1]
           :  !llvm.ptr to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// -----
// CHECK-LABEL: func @view_with_cache_mode
func.func @view_with_cache_mode(%in:  !llvm.ptr, %offset: index)
    -> memref<10x?xf32, strided<[?, 1], offset: ?>> {
  %out = aux.view %in to
           offset: [%offset], sizes: [10, 10], strides: [1, 1] {cache_mode = "cmnormal"}
           :  !llvm.ptr to memref<10x?xf32, strided<[?, 1], offset: ?>>
  return %out : memref<10x?xf32, strided<[?, 1], offset: ?>>
}

// -----
func.func @optimization_barrier(%arg0: tensor<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: index, %arg3: !llvm.ptr) {
  // CHECK: aux.optimization_barrier %arg0
  // CHECK: aux.optimization_barrier %arg1
  // CHECK: aux.optimization_barrier %arg2
  // CHECK: aux.optimization_barrier %arg3
  aux.optimization_barrier %arg0 : tensor<2x2xf32>
  aux.optimization_barrier %arg1 : memref<2x2xf32>
  aux.optimization_barrier %arg2 : index
  aux.optimization_barrier %arg3 : !llvm.ptr
  return
}

// -----
// CHECK-LABEL: @print_memref
// CHECK-COUNT-2: aux.print
func.func @print_memref(%arg0: memref<?xi8>, %arg1: memref<128xi8>) {
  aux.print(%arg0 : memref<?xi8>)
  aux.print(%arg1 : memref<128xi8>) {format = "%d"}
  return
}

// -----
// CHECK-LABEL: @print_tensor
// CHECK-COUNT-2: aux.print
func.func @print_tensor(%arg0: tensor<128xi8>, %arg1: tensor<64xi8>) ->  tensor<128xi8> {
  %1 = aux.print(%arg0 : tensor<128xi8>) -> (tensor<128xi8>)
  %2 = aux.print(%arg0 : tensor<128xi8>) {format = "%d"} -> (tensor<128xi8>)
  return %2 : tensor<128xi8>
}

// -----
// CHECK-LABEL: @print_only_format
// CHECK: aux.scalar.print
func.func @print_only_format() {
  aux.scalar.print {format = "print test"}
  return
}

// -----
// CHECK-LABEL: @print_scalar
// CHECK-COUNT-2: aux.scalar.print
func.func @print_scalar(%arg0: f32, %arg1: i32) {
  aux.scalar.print(%arg0 : f32) 
  aux.scalar.print(%arg1 : i32) {format = "%d"} 
  return 
}
