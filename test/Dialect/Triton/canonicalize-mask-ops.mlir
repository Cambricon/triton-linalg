// RUN: triton-linalg-opt --canonicalize-triton %s -split-input-file| FileCheck %s

// The case1 is as follows:
// mask non-mask               non-mask non-mask
//   \  /                           \    /
//   andi  non-mask    ======>  mask andi
//     \  /                       \  /
//     andi                       andi
func.func @case1(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i32, %arg3: index) -> tensor<64x64xi1> {
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
  %16 = arith.cmpf oge, %arg1, %arg0 : tensor<64x64xf32>
  %17 = arith.andi %15, %16 : tensor<64x64xi1>
  return %17 : tensor<64x64xi1>
}
// CHECK:      %[[MASK:.*]] = arith.andi
// CHECK-NEXT: %[[NONMASK0:.*]] = arith.cmpf olt
// CHECK-NEXT: %[[NONMASK1:.*]] = arith.cmpf oge
// CHECK-NEXT: %[[NONMASKAND:.*]] = arith.andi %[[NONMASK0]], %[[NONMASK1]] : tensor<64x64xi1>
// CHECK-NEXT: arith.andi %[[MASK]], %[[NONMASKAND]] : tensor<64x64xi1>

// -----
// The case2 is as follows:
// mask non-mask         mask mask
//   \  /                  \   /
//   andi mask  ======>    andi non-mask
//     \  /                  \   /
//     andi                  andi
func.func @case2(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i32, %arg3: index) -> tensor<64x64xi1> {
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
  %13 = arith.cmpf olt, %arg1, %arg0 : tensor<64x64xf32>
  %14 = arith.andi %12, %13 : tensor<64x64xi1>
  %15 = arith.andi %14, %7 : tensor<64x64xi1>
  return %15 : tensor<64x64xi1>
}
// CHECK:       %[[MASK0:.*]] = tt.broadcast
// CHECK:       %[[MASK1:.*]] = tt.broadcast
// CHECK-NEXT:  %[[NONMASK:.*]] = arith.cmpf
// CHECK-NEXT:  %[[MASKAND:.*]] = arith.andi %[[MASK1]], %[[MASK0]] : tensor<64x64xi1>
// CHECK-NEXT:  arith.andi %[[MASKAND]], %[[NONMASK]] : tensor<64x64xi1>


// -----
// The case3 is as follows:
// non-mask mask               mask non-mask
//       \  /    ======>         \   /
//       andi                     andi
func.func @case3(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: i32, %arg3: index) -> tensor<64x64xi1> {
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
  %15 = arith.andi %14, %13 : tensor<64x64xi1>
  return %15 : tensor<64x64xi1>
}
// CHECK:       %[[MASK:.*]] = arith.andi
// CHECK-NEXT:  %[[NONMASK:.*]] = arith.cmpf
// CHECK-NEXT:  arith.andi %[[MASK]], %[[NONMASK]] : tensor<64x64xi1>
