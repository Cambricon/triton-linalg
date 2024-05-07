//===- ArithToLinalg.h - Arith to Linalg conversion--------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
#define TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
#include <memory>
namespace mlir {
class Pass;
class RewritePatternSet;
namespace triton {

void populateArithToLinalgPatterns(RewritePatternSet &patterns);

/// Create a pass to convert a subset of arith ops to linalg.
std::unique_ptr<Pass> createArithToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
