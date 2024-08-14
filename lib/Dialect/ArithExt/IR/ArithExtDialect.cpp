//===- ArithExtDialect.cpp - Dialect for ArithExt implementation *- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/InliningUtils.h"
#include "triton-linalg/Dialect/ArithExt/IR/ArithExt.h"

using namespace mlir;
using namespace mlir::triton::arith_ext;

#include "triton-linalg/Dialect/ArithExt/IR/ArithExtOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with arith
/// operations.
struct ArithInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within arith ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::triton::arith_ext::ArithExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-linalg/Dialect/ArithExt/IR/ArithExtOps.cpp.inc"
      >();
  addInterfaces<ArithInlinerInterface>();
}
