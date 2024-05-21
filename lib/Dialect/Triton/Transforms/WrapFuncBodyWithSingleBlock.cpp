//===- WrapFuncBodyWithSingleBlock.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <iterator>
#include <memory>

#include "triton-linalg/Dialect/Triton/Transforms/PassDetail.h" // IWYU pragma: keep
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // IWYU pragma: keep
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"

using namespace mlir;
using namespace triton;

/// Encapsulate FunctionOpInterface with multiple blocks to op with a single
/// block.
static void encapsulateMultiBlock(FunctionOpInterface funcOp) {
  Region &body = funcOp.getFunctionBody();
  if (body.hasOneBlock())
    return;

  auto loc = funcOp.getLoc();
  auto *ctx = funcOp.getContext();
  OpBuilder builder(funcOp.getContext());
  // Add a new entry block to the func op.
  auto &entryBlock = body.front();
  auto blockArgTypes = entryBlock.getArgumentTypes();
  SmallVector<Location> blockArgLocs(blockArgTypes.size(), loc);
  Block *newBlock =
      builder.createBlock(&body, body.begin(), blockArgTypes, blockArgLocs);

  // Add scf.execute_region to the entry block.
  builder.setInsertionPointToStart(newBlock);
  FunctionType funcType = funcOp.getFunctionType().cast<FunctionType>();
  auto containerOp =
      builder.create<scf::ExecuteRegionOp>(loc, funcType.getResults());
  auto &containerRegion = containerOp.getRegion();

  // Move original blocks to the region of scf.execute_region.
  auto &blocksToMove = body.getBlocks();
  containerRegion.getBlocks().splice(containerRegion.end(), blocksToMove,
                                     std::next(blocksToMove.begin()),
                                     blocksToMove.end());

  // Since the entry block of scf.execute_region must not have arguments,
  // remove block arguments and replace the uses with outer block arguments.
  auto &containerEntryBlock = containerRegion.getBlocks().front();
  for (auto &arg : containerEntryBlock.getArguments()) {
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      auto newValue = newBlock->getArgument(arg.getArgNumber());
      use.getOwner()->setOperand(use.getOperandNumber(), newValue);
    }
  }
  containerEntryBlock.eraseArguments(0, containerEntryBlock.getNumArguments());

  // Replace return with scf.yield.
  SmallVector<Operation *> returnOps;
  containerOp.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::ReturnLike>() &&
        op->getParentOp() == containerOp.getOperation())
      returnOps.push_back(op);
  });
  assert(!returnOps.empty());

  // Since the bufferization of scf.execute_region only supports unique
  // scf.yield, we add a yield block to sink all scf.yield here.
  auto &yieldBlock = containerRegion.emplaceBlock();
  auto operandTypes = returnOps.front()->getOperandTypes();
  auto unknownLoc = UnknownLoc::get(ctx);
  SmallVector<Location> unknownLocs(operandTypes.size(), unknownLoc);
  yieldBlock.addArguments(operandTypes, unknownLocs);
  builder.setInsertionPointToEnd(&yieldBlock);
  builder.create<scf::YieldOp>(unknownLoc, yieldBlock.getArguments());

  for (auto *returnOp : returnOps) {
    builder.setInsertionPoint(returnOp);
    builder.create<cf::BranchOp>(returnOp->getLoc(), &yieldBlock,
                                 returnOp->getOperands());
  }

  // Since we do not know the concrete return op type, we just keep the last
  // return op, and update it's operands.
  Operation *terminator = returnOps.pop_back_val();
  for (auto *returnOp : returnOps)
    returnOp->erase();

  terminator->setOperands(containerOp.getResults());
  builder.setInsertionPointToEnd(&body.back());
  terminator->remove();
  builder.insert(terminator);
}

namespace {

/// Pass that wrap triton function body with a single block by moving
/// multi-block function body into a `scf.execute_region`.
/// For example:
///
/// ```mlir
///   tt.func @foo(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
///     cf.cond_br %arg0, ^bb1, ^bb2
///   ^bb1:
///     tt.return %arg1: i32
///   ^bb2:
///     tt.return %arg2: i32
///   }
/// ```
///
/// Is converted to:
///
/// ```mlir
///   tt.func @foo(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
///     %0 = scf.execute_region -> i32 {
///       cf.cond_br %arg0, ^bb1, ^bb2
///     ^bb1:
///       cb.br ^bb3(%arg1 : i32)
///     ^bb2:
///       cb.br ^bb3(%arg2 : i32)
///     ^bb3(%1 : i32):
///       scf.yield %1 : i32
///     }
///     tt.return %0 : i32
///   }
/// ```
struct WrapFuncBodyWithSingleBlockPass
    : public WrapFuncBodyWithSingleBlockBase<WrapFuncBodyWithSingleBlockPass> {
  void runOnOperation() override {
    getOperation()->walk(
        [&](FunctionOpInterface func) { encapsulateMultiBlock(func); });
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
std::unique_ptr<Pass> createWrapFuncBodyWithSingleBlockPass() {
  return std::make_unique<WrapFuncBodyWithSingleBlockPass>();
}

} // namespace triton
} // namespace mlir
