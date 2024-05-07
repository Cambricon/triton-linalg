// RUN: triton-linalg-opt --wrap-func-body-with-single-block -split-input-file %s | FileCheck %s

tt.func @wrap_multi_block_triton_func(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  tt.return %arg1: i32
^bb2:
  tt.return %arg2: i32
}
// CHECK-LABEL:   tt.func @wrap_multi_block_triton_func(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                          %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = scf.execute_region -> i32 {
// CHECK:             cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb3(%[[VAL_1]] : i32)
// CHECK:           ^bb2:
// CHECK:             cf.br ^bb3(%[[VAL_2]] : i32)
// CHECK:           ^bb3(%[[VAL_4:.*]]: i32):
// CHECK:             scf.yield %[[VAL_4]] : i32
// CHECK:           }
// CHECK:           tt.return %[[VAL_3]] : i32
// CHECK:         }

// -----
func.func @wrap_multi_block_func(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  return %arg1: i32
^bb2:
  return %arg2: i32
}
// CHECK-LABEL:   func.func @wrap_multi_block_func(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                     %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                     %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_3:.*]] = scf.execute_region -> i32 {
// CHECK:             cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb3(%[[VAL_1]] : i32)
// CHECK:           ^bb2:
// CHECK:             cf.br ^bb3(%[[VAL_2]] : i32)
// CHECK:           ^bb3(%[[VAL_4:.*]]: i32):
// CHECK:             scf.yield %[[VAL_4]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

// -----
tt.func @ignore_single_block_func() {
  tt.return
}
// CHECK-LABEL:   tt.func @ignore_single_block_func
// CHECK-NOT:     scf.execute_region
