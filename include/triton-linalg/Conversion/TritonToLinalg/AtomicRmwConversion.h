//===- AtomicRmwConversion.h - atomicRmw op conversion-----------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_ATOMICRMWCONVERSION_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_ATOMICRMWCONVERSION_H

namespace mlir {
class DataFlowSolver;
class RewritePatternSet;
namespace triton {
class TritonLinalgTypeConverter;

void populateTritonAtomicRmwToLinalgPatterns(
    RewritePatternSet &patterns, TritonLinalgTypeConverter &converter,
    mlir::DataFlowSolver &solver);

} // namespace triton
} // namespace mlir
#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_ATOMICRMWCONVERSION_H
