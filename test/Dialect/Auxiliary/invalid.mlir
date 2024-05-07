// RUN: triton-linalg-opt %s -split-input-file -verify-diagnostics --allow-unregistered-dialect

// -----
func.func @view_too_many_offsets(%in: !llvm.ptr) {
  // expected-error @+1 {{expected 1 offset values}}
  %out = aux.view %in to
           offset: [0, 0], sizes: [10, 10], strides: [10, 1]
           : !llvm.ptr to memref<10x10xf32, strided<[10, 1], offset: 0>>
  return
}

// -----

func.func @view_incompatible_memory_space(%in: !llvm.ptr) {
  // expected-error @+1 {{different memory spaces specified}}
  %out = aux.view %in to
           offset: [0], sizes: [10], strides: [1]
         : !llvm.ptr to memref<10xi32, strided<[1], offset: 0>, 2>
  return
}

// -----

func.func @view_offset_mismatch(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with offset = 2 instead of 1}}
  %out = aux.view %in to
           offset: [1], sizes: [10], strides: [1]
         : !llvm.ptr to memref<10xf32, strided<[1], offset: 2>>
  return
}

// -----

func.func @view_size_mismatch(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with size = 10 instead of 1 in dim = 0}}
  %out = aux.view %in to
           offset: [0], sizes: [10], strides: [1]
         : !llvm.ptr to memref<1xf32, strided<[1], offset: 0>>
  return
}

// -----

func.func @view_offset_mismatch(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with stride = 2 instead of 1 in dim = 0}}
  %out = aux.view %in to
           offset: [2], sizes: [10], strides: [2]
         : !llvm.ptr to memref<10xf32, strided<[1], offset: 2>>
  return
}

// -----

func.func @view_no_map_but_offset(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with offset = 0 instead of 2}}
  %out = aux.view %in to offset: [2], sizes: [10], strides: [1]
         : !llvm.ptr to memref<10xf32>
  return
}

// -----

func.func @view_no_map_but_stride(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with stride = 10 instead of 1 in dim = 0}}
  %out = aux.view %in to offset: [0], sizes: [10], strides: [10]
         : !llvm.ptr to memref<10xf32>
  return
}

// -----

func.func @view_no_map_but_strides(%in: !llvm.ptr) {
  // expected-error @+1 {{expected result type with stride = 42 instead of 10 in dim = 0}}
  %out = aux.view %in to
           offset: [0], sizes: [9, 10], strides: [42, 1]
         : !llvm.ptr to memref<9x10xf32>
  return
}

// -----

func.func @view_invalid_cache_mode(%in: !llvm.ptr) {
  // expected-error @+1 {{expected legal cache mode but found cmlock}}
  %out = aux.view %in to
           offset: [0], sizes: [9, 42], strides: [42, 1] {cache_mode = "cmlock"}
         : !llvm.ptr to memref<9x42xf32>
  return
}

// -----
func.func @print_wrong_num(%arg0: tensor<128xi8>, %arg1: tensor<64xi8>) -> tensor<128xi8> {
  // expected-error @+1 {{'aux.print' op only accepts 1 input operand atmost!}}
  %1 = aux.print(%arg0, %arg1 : tensor<128xi8>, tensor<64xi8>) -> (tensor<128xi8>)
  return
}

// -----
func.func @print_wrong_valid_fmt_num(%arg0: tensor<128xi8>) -> tensor<128xi8> {
  // expected-error @+1 {{'aux.print' op Expected valid format num is 1 but now it is 2}}
  %1 = aux.print(%arg0 : tensor<128xi8>) {format = "%d, %f"} -> (tensor<128xi8>)
  return
}

// -----
func.func @scalar_print_wrong_valid_fmt_num(%arg0: f32) {
  // expected-error @+1{{'aux.scalar.print' op Operands num 1 need equal to valid fmt num 2}}
  aux.scalar.print(%arg0 : f32) {format = "%d, %f"}
  return
}
