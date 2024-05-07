//===- LoadStoreConversion.h - Load/store op convension ---------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_LOADSTORECONVERSION_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_LOADSTORECONVERSION_H

namespace mlir {
class DataFlowSolver;
class RewritePatternSet;
namespace triton {
class TritonLinalgTypeConverter;

void populateTritonLoadStoreToLinalgPatterns(
    RewritePatternSet &patterns, TritonLinalgTypeConverter &converter,
    mlir::DataFlowSolver &solver);
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_LOADSTORECONVERSION_H
