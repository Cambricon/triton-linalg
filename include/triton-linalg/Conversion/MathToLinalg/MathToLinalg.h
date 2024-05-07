//===- MathToLinalg.h - Math to Linalg conversion--------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_MATHTOLINALG_MATHTOLINALG_H
#define TRITON_LINALG_CONVERSION_MATHTOLINALG_MATHTOLINALG_H
#include <memory>
namespace mlir {
class Pass;
class RewritePatternSet;
namespace triton {

void populateMathToLinalgPatterns(RewritePatternSet &patterns);

/// Create a pass to convert a subset of math ops to linalg.
std::unique_ptr<Pass> createMathToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_MATHTOLINALG_MATHTOLINALG_H
