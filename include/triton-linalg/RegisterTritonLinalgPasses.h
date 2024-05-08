#pragma once
#include "triton-linalg/Conversion/Passes.h"
#include "triton-linalg/Transforms/Passes.h"
#include "triton-linalg/Dialect/Arith/Passes.h"

inline void registerTritonLinalgPasses() {
  ::mlir::triton::arith_ext::registerArithExtPasses();
  ::mlir::triton::registerTritonLinalgConversionPasses();
  ::mlir::triton::registerTritonLinalgTransformsPasses();
}
