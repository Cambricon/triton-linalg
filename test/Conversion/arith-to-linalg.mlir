// RUN: triton-linalg-opt -convert-arith-to-linalg -split-input-file %s | FileCheck %s

// -----

func.func @const_valid_float(%arg0: tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32> {
  // CHECK:  %cst = arith.constant 0.000000e+00 : f32
  // CHECK:  tensor.empty
  // CHECK:  linalg.fill
  // CHECK:  linalg.map
  %cst = arith.constant dense<0.000000e+00> : tensor<1x16x128x128xf32>
  %0 = tensor.empty() : tensor<1x16x128x128xf32>
  %mapped = linalg.map { arith.maximumf } ins(%arg0, %cst : tensor<1x16x128x128xf32>, tensor<1x16x128x128xf32>) outs(%0 : tensor<1x16x128x128xf32>)
  return %mapped: tensor<1x16x128x128xf32>
}

// -----

func.func @const_valid_int(%arg0: tensor<1x16x128x128xi32>) -> tensor<1x16x128x128xi32> {
  // CHECK:  %c0_i32 = arith.constant 0 : i32
  // CHECK:  tensor.empty
  // CHECK:  linalg.fill
  // CHECK:  linalg.map
  %cst = arith.constant dense<0> : tensor<1x16x128x128xi32>
  %0 = tensor.empty() : tensor<1x16x128x128xi32>
  %mapped = linalg.map { arith.addi } ins(%arg0, %cst : tensor<1x16x128x128xi32>, tensor<1x16x128x128xi32>) outs(%0 : tensor<1x16x128x128xi32>)
  return %mapped: tensor<1x16x128x128xi32>
}

// -----
func.func @arith_addi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.addi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_subi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.subi {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.subi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_muli(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.muli {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.muli %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_divui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.divui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.divui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_divsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.divsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.divsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_ceildivui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.ceildivui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.ceildivui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_ceildivsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.ceildivsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.ceildivsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_floordivsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.floordivsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.floordivsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_remui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.remui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.remui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_remsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.remsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.remsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_andi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.andi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.andi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_ori(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.ori } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.ori %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_xori(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.xori } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.xori %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_shli(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.shli {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.shli %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_shrui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.shrui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.shrui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_shrsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.shrsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.shrsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_negf(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.negf } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.negf %arg0 : tensor<128xf32>
  return
}

// -----
func.func @arith_addf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.addf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_subf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.subf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.subf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith.maximumf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.maximumf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.maximumf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_maxsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.maxsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.maxsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_maxui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.maxui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.maxui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith.minimumf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.minimumf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.minimumf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_minsi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.minsi } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.minsi %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_minui(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.minui } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi32>)
  %0 = arith.minui %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_mulf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.mulf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.mulf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_divf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.divf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.divf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_remf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.remf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.remf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_extui(%arg0: tensor<128xi1>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi8>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.extui } ins(%arg0 : tensor<128xi1>) outs(%[[INIT]] : tensor<128xi8>)
  %0 = arith.extui %arg0 : tensor<128xi1> to tensor<128xi8>
  return
}

// -----
func.func @arith_index_cast(%arg0: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xindex>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.index_cast } ins(%arg0 : tensor<128xi32>) outs(%[[INIT]] : tensor<128xindex>)
  %0 = arith.index_cast %arg0 : tensor<128xi32> to tensor<128xindex>
  return
}

// -----
func.func @arith_extsi(%arg0: tensor<128xi1>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi8>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.extsi } ins(%arg0 : tensor<128xi1>) outs(%[[INIT]] : tensor<128xi8>)
  %0 = arith.extsi %arg0 : tensor<128xi1> to tensor<128xi8>
  return
}

// -----
func.func @arith_extf(%arg0: tensor<128xf16>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.extf } ins(%arg0 : tensor<128xf16>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.extf %arg0 : tensor<128xf16> to tensor<128xf32>
  return
}

// -----
func.func @arith_trunci(%arg0: tensor<128xi8>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi1>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.trunci } ins(%arg0 : tensor<128xi8>) outs(%[[INIT]] : tensor<128xi1>)
  %0 = arith.trunci %arg0 : tensor<128xi8> to tensor<128xi1>
  return
}

// -----
func.func @arith_truncf(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf16>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.truncf } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf16>)
  %0 = arith.truncf %arg0 : tensor<128xf32> to tensor<128xf16>
  return
}

// -----
func.func @arith_uitofp(%arg0: tensor<128xi8>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.uitofp } ins(%arg0 : tensor<128xi8>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.uitofp %arg0 : tensor<128xi8> to tensor<128xf32>
  return
}

// -----
func.func @arith_sitofp(%arg0: tensor<128xi8>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.sitofp } ins(%arg0 : tensor<128xi8>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.sitofp %arg0 : tensor<128xi8> to tensor<128xf32>
  return
}

// -----
func.func @arith_fptoui(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi8>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.fptoui } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xi8>)
  %0 = arith.fptoui %arg0 : tensor<128xf32> to tensor<128xi8>
  return
}

// -----
func.func @arith_fptosi(%arg0: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi8>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.fptosi } ins(%arg0 : tensor<128xf32>) outs(%[[INIT]] : tensor<128xi8>)
  %0 = arith.fptosi %arg0 : tensor<128xf32> to tensor<128xi8>
  return
}

// -----
func.func @arith_cmpi(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi1>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.cmpi {predicate = 0 : i64} } ins(%arg0, %arg1 : tensor<128xi32>, tensor<128xi32>) outs(%[[INIT]] : tensor<128xi1>)
  %0 = arith.cmpi "eq", %arg0, %arg1 : tensor<128xi32>
  return
}

// -----
func.func @arith_cmpf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xi1>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.cmpf {predicate = 8 : i64} } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xi1>)
  %0 = arith.cmpf "ueq", %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith_select(%arg0: tensor<128xi1>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.select } ins(%arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128xf32>
  return
}

// -----
func.func @arith_select_scalar_condition(%arg0: i1, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) {
  // CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<128xi1>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%arg0 : i1) outs(%[[INIT1]] : tensor<128xi1>) -> tensor<128xi1>
  // CHECK: %[[INIT2:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.select } ins(%[[FILL]], %arg1, %arg2 : tensor<128xi1>, tensor<128xf32>, tensor<128xf32>) outs(%[[INIT2]] : tensor<128xf32>)
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<128xf32>
  return
}


// -----
func.func @arith_select_scalar_condition_dynamic_output(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?xf32>
  // CHECK: %[[INIT1:.*]] = tensor.empty(%[[DIM]]) : tensor<?xi1>
  // CHECK: %[[FILL:.*]] = linalg.fill ins(%arg0 : i1) outs(%[[INIT1]] : tensor<?xi1>) -> tensor<?xi1>
  // CHECK: %[[INIT2:.*]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.select } ins(%[[FILL]], %arg1, %arg2 : tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) outs(%[[INIT2]] : tensor<?xf32>)
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<?xf32>
  return
}

// -----
func.func @arith_addi_scalar(%arg0: i32, %arg1: i32) {
  // CHECK: arith.addi
  // CHECK-NOT: linalg.map
  %0 = arith.addi %arg0, %arg1 : i32
  return
}

// -----
func.func @arith_addi_dynamic(%arg0: tensor<128x?xi32>, %arg1: tensor<128x?xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[DYNAMIC_DIM:.*]] = tensor.dim %arg0, %[[CST]] : tensor<128x?xi32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DYNAMIC_DIM]]) : tensor<128x?xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.addi {overflowFlags = #arith.overflow<none>} } ins(%arg0, %arg1 : tensor<128x?xi32>, tensor<128x?xi32>) outs(%[[INIT]] : tensor<128x?xi32>)
  %0 = arith.addi %arg0, %arg1 : tensor<128x?xi32>
  return
}

// -----
func.func @arith_bitcast_scalar(%arg0: i32) {
  // CHECK: arith.bitcast %arg0 : i32 to f32
  // CHECK-NOT: linalg.map
  %0 = arith.bitcast %arg0 : i32 to f32
  return
}


// -----
func.func @arith_bitcast_static(%arg0: tensor<128x128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128x128xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.bitcast } ins(%arg0 : tensor<128x128xf32>) outs(%[[INIT]] : tensor<128x128xi32>)
  %0 = arith.bitcast %arg0 : tensor<128x128xf32> to tensor<128x128xi32>
  return
}

// -----
func.func @arith_bitcast_partial_static(%arg0: tensor<128x?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[D0:.*]] =  tensor.dim %arg0, %[[CST]] : tensor<128x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]]) : tensor<128x?xi32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.bitcast } ins(%arg0 : tensor<128x?xf32>) outs(%[[INIT]] : tensor<128x?xi32>)
  %0 = arith.bitcast %arg0 : tensor<128x?xf32> to tensor<128x?xi32>
  return
}

// -----
func.func @arith_bitcast_dynamic(%arg0: tensor<?x?xf32>) {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[D0:.*]] = tensor.dim %arg0, %[[CST0]] : tensor<?x?xf32>
  // CHECK-NEXT: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[D1:.*]] = tensor.dim %arg0, %[[CST1]] : tensor<?x?xf32>
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xi32>
  // CHECK-NEXT: %[[MAPPED:.*]] = linalg.map { arith.bitcast } ins(%arg0 : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xi32>)
  %0 = arith.bitcast %arg0 : tensor<?x?xf32> to tensor<?x?xi32>
  return
}

// -----
func.func @arith.minnumf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.minnumf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.minnumf %arg0, %arg1 : tensor<128xf32>
  return
}

// -----
func.func @arith.maxnumf(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[MAPPED:.*]] = linalg.map { arith.maxnumf } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>)
  %0 = arith.maxnumf %arg0, %arg1 : tensor<128xf32>
  return
}
