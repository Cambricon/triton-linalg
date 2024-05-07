//===- MathDialect.cpp - MLIR dialect for Math implementation -------------===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/InliningUtils.h"
#include "triton-linalg/Dialect/MathExt/IR/Math.h"

using namespace mlir;
using namespace mlir::math_ext;

#include "triton-linalg/Dialect/MathExt/IR/MathExtOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::math_ext::MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-linalg/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
  addInterfaces<MathInlinerInterface>();
}
