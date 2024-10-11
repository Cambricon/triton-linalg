//===- LinalgExtInterface.cpp -[Desc]----------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   // IWYU pragma: keep
#include "mlir/Dialect/Linalg/IR/Linalg.h" // IWYU pragma: keep
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::triton;

namespace mlir {
class OpBuilder;
namespace triton {
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtInterface.cpp.inc" // IWYU pragma: export
} // namespace triton
} // namespace mlir
LogicalResult triton::detail::verifyLinalgExtOpInterface(Operation *op) {
  if (isa<mlir::DestinationStyleOpInterface>(op) && isa<TilingInterface>(op)) {
    return success();
  }
  return op->emitOpError(
      "expected linalgext op is also a dest style op with tiling interface");
}

LogicalResult LinalgExtOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return llvm::cast<LinalgOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}
