//===- Utils.cpp - Analysis utilities ---------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "triton/CodeGen/Analysis/Utils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
namespace triton {
namespace dataflow {

std::unique_ptr<mlir::DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<mlir::DataFlowSolver>();
  solver->load<mlir::dataflow::DeadCodeAnalysis>();
  solver->load<mlir::dataflow::SparseConstantPropagation>();
  return solver;
}

} // namespace dataflow
} // namespace triton
} // namespace mlir
