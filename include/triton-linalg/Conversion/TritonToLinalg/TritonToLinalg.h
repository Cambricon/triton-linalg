//===- TritonToLinalg.h - Triton to Linalg dialect convension ---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class ConversionTarget;
class DataFlowSolver;
namespace triton {
class TritonLinalgTypeConverter;

void populateAllTritonToLinalgPattern(RewritePatternSet &patterns,
                                      TritonLinalgTypeConverter &converter,
                                      ConversionTarget &target,
                                      mlir::DataFlowSolver &solver);

/// Create a pass to convert a subset of Triton ops to Linalg.
std::unique_ptr<mlir::Pass> createTritonToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
