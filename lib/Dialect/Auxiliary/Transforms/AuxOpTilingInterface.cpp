//===- AuxOpTilingInterface.cpp -AuxOp TilingInterface impl------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <memory>
#include <stdint.h>

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/Auxiliary/Transforms/AuxOpTilingInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h" // IWYU pragma: keep
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h" // IWYU pragma: keep
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton;

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace {

struct PrintOpTilingInterface
    : public TilingInterface::ExternalModel<PrintOpTilingInterface,
                                            triton::aux::PrintOp> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return llvm::cast<DestinationStyleOpInterface>(op).getDpsInits();
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    triton::aux::PrintOp printOp = cast<triton::aux::PrintOp>(op);
    SmallVector<utils::IteratorType> loops(printOp.getInitType().getRank(),
                                           utils::IteratorType::parallel);
    return loops;
  }

  // Tile init then update.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    triton::aux::PrintOp printOp = cast<triton::aux::PrintOp>(op);
    Location loc = op->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Range> ranges;
    for (auto dim : llvm::seq<int64_t>(0, printOp.getInitType().getRank())) {
      Value ub = getDimValue(b, loc, printOp.getOperands()[0], dim);
      ranges.emplace_back(Range{zero, ub, one});
    }
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    triton::aux::PrintOp printOp = cast<triton::aux::PrintOp>(op);
    Location loc = printOp.getLoc();
    int64_t rank = printOp.getInitType().getRank();

    SmallVector<OpFoldResult> inputOffsets(offsets.begin(), offsets.end());
    SmallVector<OpFoldResult> inputSizes(sizes.begin(), sizes.end());
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);

    auto inputSlice = b.create<tensor::ExtractSliceOp>(
        loc, printOp.getOperands()[0], inputOffsets, inputSizes, strides);

    Operation *tiledOp = b.create<triton::aux::PrintOp>(
        loc, inputSlice.getType(), ValueRange{inputSlice},
        printOp.getFormatAttr());

    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    return tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &b,
                                             Location loc,
                                             ValueRange ivs) const {
    llvm_unreachable(
        "Unreachable code reached in generateScalarImplementation");
  }
};
} // namespace

void mlir::triton::aux::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, triton::aux::AuxiliaryDialect *dialect) {
        triton::aux::PrintOp::attachInterface<PrintOpTilingInterface>(*ctx);
      });
}
