//===- Utils.cpp - Utilities to support the Linalg dialect ------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the LinalgExt dialect.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdint.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"

using namespace mlir;
using namespace triton;

namespace mlir {
class Block;
class Operation;
} // namespace mlir

Operation *triton::linalg_ext::findPayloadOp(Block *body, bool initFirst) {
  if (body->getOperations().size() != 2)
    return nullptr;
  Operation &payload = body->getOperations().front();
  assert(body->getOperations().back().hasTrait<OpTrait::IsTerminator>());

  if (payload.getNumOperands() == 0 ||
      payload.getNumOperands() != body->getNumArguments())
    return nullptr;
  if (initFirst) {
    // check init
    if (payload.getOperands().back() != body->getArgument(0))
      return nullptr;
    // check rest
    for (const auto &[operand, bbArg] :
         llvm::zip(payload.getOperands(), body->getArguments().drop_front())) {
      if (bbArg != operand)
        return nullptr;
    }
  } else {
    for (const auto &[operand, bbArg] :
         llvm::zip(payload.getOperands(), body->getArguments())) {
      if (bbArg != operand)
        return nullptr;
    }
  }
  return &payload;
}

/// Check whether the reduce op is supported and get the reduction mode
/// if supported.
std::optional<ReductionMode> triton::getReductionMode(triton::ReduceOp op) {
  if (isSingleStatementReduceOpWithType<arith::AddFOp, triton::ReduceOp>(op) ||
      isSingleStatementReduceOpWithType<arith::AddIOp, triton::ReduceOp>(op))
    return ReductionMode::SUM;

  if (isSingleStatementReduceOpWithType<arith::MaxNumFOp, triton::ReduceOp>(
          op) ||
      isSingleStatementReduceOpWithType<arith::MaximumFOp, triton::ReduceOp>(
          op) ||
      isSingleStatementReduceOpWithType<arith::MaxSIOp, triton::ReduceOp>(op))
    return ReductionMode::MAX;

  if (isSingleStatementReduceOpWithType<arith::MaxUIOp, triton::ReduceOp>(op))
    return ReductionMode::UMAX;

  if (isSingleStatementReduceOpWithType<arith::MinNumFOp, triton::ReduceOp>(
          op) ||
      isSingleStatementReduceOpWithType<arith::MinimumFOp, triton::ReduceOp>(
          op) ||
      isSingleStatementReduceOpWithType<arith::MinSIOp, triton::ReduceOp>(op))
    return ReductionMode::MIN;

  if (isSingleStatementReduceOpWithType<arith::MinUIOp, triton::ReduceOp>(op))
    return ReductionMode::UMIN;

  if (isSingleStatementReduceOpWithType<arith::MulFOp, triton::ReduceOp>(op))
    return ReductionMode::PROD;

  if (isSingleStatementReduceOpWithType<arith::AndIOp, triton::ReduceOp>(op))
    return ReductionMode::AND;

  if (isSingleStatementReduceOpWithType<arith::OrIOp, triton::ReduceOp>(op))
    return ReductionMode::OR;

  if (isSingleStatementReduceOpWithType<arith::XOrIOp, triton::ReduceOp>(op))
    return ReductionMode::XOR;
  // Unsupport reduce op mode.
  return std::nullopt;
}

template <class CmpOp>
static std::optional<ReductionMode> matchArgMaxMinPatternImpl(Region *region) {
  // We're looking for an op that looks like this:
  //
  // %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
  // ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
  // -------------------------------------------------
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
  //   %13 = arith.andi %11, %12 : i1
  // -------------------------------------------------
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32
  // -------------------------------------------------
  //   %15 = arith.ori %14, %13 : i1
  // -------------------------------------------------
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   tt.reduce.return %16, %17 : f32, i32
  // }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  if (region->getNumArguments() != 4) {
    return std::nullopt;
  }

  Block &block = region->front();
  // There are 8 fixed operations within the argmaxmin region.
  if (block.getOperations().size() != 8) {
    return std::nullopt;
  }
  Operation *terminatorOp = block.getTerminator();

  //   %15 = arith.ori %14, %13 : i1
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   linalg.yield %16, %17 : f32, i32
  SmallVector<Operation *> lineOut0;
  SmallVector<int> inputIndex0 = {0, 0};
  Operation *result0 =
      UpstreamMatcher<Operation *, arith::SelectOp, arith::OrIOp>::matchLine(
          lineOut0, terminatorOp, inputIndex0, inputIndex0.size(), false);
  if (result0 == nullptr ||
      cast<arith::SelectOp>(lineOut0[1]).getTrueValue() !=
          block.getArgument(0) ||
      cast<arith::SelectOp>(lineOut0[1]).getFalseValue() !=
          block.getArgument(2)) {
    return std::nullopt;
  }

  //   %15 = arith.ori %14, %13 : i1
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   linalg.yield %16, %17 : f32, i32
  SmallVector<Operation *> lineOut1;
  SmallVector<int> inputIndex1 = {1, 0};
  Operation *result1 =
      UpstreamMatcher<Operation *, arith::SelectOp, arith::OrIOp>::matchLine(
          lineOut1, terminatorOp, inputIndex1, inputIndex1.size(), false);
  if (result1 == nullptr || lineOut1[2] != lineOut0[2] ||
      cast<arith::SelectOp>(lineOut1[1]).getTrueValue() !=
          block.getArgument(1) ||
      cast<arith::SelectOp>(lineOut1[1]).getFalseValue() !=
          block.getArgument(3)) {
    return std::nullopt;
  }

  auto *oriOp = lineOut0[2];
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32
  //   %15 = arith.ori %14, %13 : i1
  SmallVector<Operation *> lineOut2;
  SmallVector<int> inputIndex2 = {0};
  Operation *result2 = UpstreamMatcher<arith::OrIOp, CmpOp>::matchLine(
      lineOut2, oriOp, inputIndex2, inputIndex2.size(), false);
  if (result2 == nullptr ||
      cast<CmpOp>(lineOut2[1]).getLhs() != block.getArgument(0) ||
      cast<CmpOp>(lineOut2[1]).getRhs() != block.getArgument(2)) {
    return std::nullopt;
  }

  //   %13 = arith.andi %11, %12 : i1
  //   %15 = arith.ori %14, %13 : i1
  SmallVector<Operation *> lineOut3;
  SmallVector<int> inputIndex3 = {1};
  Operation *result3 = UpstreamMatcher<arith::OrIOp, arith::AndIOp>::matchLine(
      lineOut3, oriOp, inputIndex3, inputIndex3.size(), false);
  if (result3 == nullptr) {
    return std::nullopt;
  }

  auto *andiOp = lineOut3[1];
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
  //   %13 = arith.andi %11, %12 : i1
  SmallVector<Operation *> lineOut4;
  SmallVector<int> inputIndex4 = {0};
  Operation *result4 = UpstreamMatcher<arith::AndIOp, CmpOp>::matchLine(
      lineOut4, andiOp, inputIndex4, inputIndex4.size(), false);
  if (result4 == nullptr ||
      cast<CmpOp>(lineOut4[1]).getLhs() != block.getArgument(0) ||
      cast<CmpOp>(lineOut4[1]).getRhs() != block.getArgument(2)) {
    return std::nullopt;
  }

  if constexpr (std::is_same_v<CmpOp, arith::CmpFOp>) {
    if (cast<CmpOp>(lineOut4[1]).getPredicate() != arith::CmpFPredicate::OEQ)
      return std::nullopt;
  } else if constexpr (std::is_same_v<CmpOp, arith::CmpIOp>) {
    if (cast<CmpOp>(lineOut4[1]).getPredicate() != arith::CmpIPredicate::eq)
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
  //   %13 = arith.andi %11, %12 : i1
  SmallVector<Operation *> lineOut5;
  SmallVector<int> inputIndex5 = {1};
  Operation *result5 = UpstreamMatcher<arith::AndIOp, arith::CmpIOp>::matchLine(
      lineOut5, andiOp, inputIndex5, inputIndex5.size(), false);
  if (result5 == nullptr ||
      cast<arith::CmpIOp>(lineOut5[1]).getPredicate() !=
          arith::CmpIPredicate::slt ||
      cast<arith::CmpIOp>(lineOut5[1]).getLhs() != block.getArgument(1) ||
      cast<arith::CmpIOp>(lineOut5[1]).getRhs() != block.getArgument(3)) {
    return std::nullopt;
  }

  auto cmpfOp = cast<CmpOp>(lineOut2[1]);
  if constexpr (std::is_same_v<CmpOp, arith::CmpFOp>) {
    if (cmpfOp.getPredicate() == arith::CmpFPredicate::OGT) {
      return ReductionMode::ARGMAX;
    }
    if (cmpfOp.getPredicate() == arith::CmpFPredicate::OLT) {
      return ReductionMode::ARGMIN;
    }
  }
  if constexpr (std::is_same_v<CmpOp, arith::CmpIOp>) {
    if (cmpfOp.getPredicate() == arith::CmpIPredicate::sgt) {
      return ReductionMode::ARGMAX;
    }
    if (cmpfOp.getPredicate() == arith::CmpIPredicate::slt) {
      return ReductionMode::ARGMIN;
    }
  }

  return std::nullopt;
}

/// Check whether the reduce op can convert to argmax/min operation.
std::optional<ReductionMode> triton::matchArgMaxMinPattern(Region *region) {
  auto result = matchArgMaxMinPatternImpl<arith::CmpFOp>(region);
  if (result != std::nullopt)
    return result;
  return matchArgMaxMinPatternImpl<arith::CmpIOp>(region);
}

/// Identify the pattern of the reduce operator.
std::optional<ReductionMode>
triton::reducePatternRecognition(triton::ReduceOp op) {
  auto mode = getReductionMode(op);
  if (mode.has_value()) {
    return mode;
  }
  mode = matchArgMaxMinPattern(&op.getRegion());
  if (mode.has_value()) {
    return mode;
  }

  return std::nullopt;
}
