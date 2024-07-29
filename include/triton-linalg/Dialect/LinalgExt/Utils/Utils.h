//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H
#define TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H
#include <optional>
#include <stdint.h>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <initializer_list>
namespace mlir {
class Block;
class MLIRContext;
class RewriterBase;
class Operation;

namespace linalg {
class ReduceOp;
} // namespace linalg

namespace triton {
namespace linalg_ext {
/// Retrieve the operation from the body, if it is the only one (except
/// yield) and if it gets the same amount of arguments as the body does.
/// If initFirst flag is enabled, we check that init takes the first position in
/// operands of payload.
Operation *findPayloadOp(Block *body, bool initFirst = false);
} // namespace linalg_ext

constexpr static int ANY_INDEX = -1;
template <typename... T0> struct UpstreamMatcher;

template <typename T1, typename... T2> struct UpstreamMatcher<T1, T2...> {

  /// Matches a line-shaped net. Multiple inputs are allowed, but multiple
  /// outputs are not.
  ///
  /// example:
  ///                      OP1          /|\  (where Values flow out)
  ///                       |            |
  ///                      OP2           |
  ///                    /  |  \         |
  ///                 OP5  OP4  OP3      |   (where Values flow in)
  ///
  /// Function output:
  /// - `lineOut`: If the match is successful, this will contain the matched
  ///   operations (oplist).
  ///
  /// Inputs:
  /// - `inputOp`: One of the operations in the graph to be matched. Since the
  ///   matching is performed from back to front, this is the last operation
  ///   in the graph.
  /// - `inputIndex`: Specifies the operand indices, indicating which operand
  ///   of `OP2` corresponds to `OP1` and which operand of `OP3` corresponds
  ///   to `OP2`.
  /// - `originLength`: The number of operations to be matched in this pass.
  ///
  /// Example code for match {OP1, OP2, OP3}:
  /// SmallVector<Operation*> lineOut;
  /// SmallVector<int> inputIndex = {0, 2};
  /// Operation *result = UpstreamMatcher::matchLine<OP1, OP2, OP3>(
  ///    lineOut, OP1Operation, inputIndex, inputIndex.size(), false);
  /// if (result == nullptr) {
  ///    std::cout << "Check failed" << std::endl;
  /// }
  ///
  /// Example code for match {OP1, OP2, OP4}:
  /// SmallVector<Operation*> lineOut;
  /// SmallVector<int> inputIndex = {0, 1};
  /// Operation *result = UpstreamMatcher::matchLine<OP1, OP2, OP4>(
  ///    lineOut, OP1Operation, inputIndex, inputIndex.size(), false);
  /// if (result == nullptr) {
  ///    std::cout << "Check failed" << std::endl;
  /// }
  ///
  /// Example code for match {OP1, OP2, OP5}:
  /// SmallVector<Operation*> lineOut;
  /// SmallVector<int> inputIndex = {0, 0};
  /// Operation *result = UpstreamMatcher::matchLine<OP1, OP2, OP5>(
  ///    lineOut, OP1Operation, inputIndex, inputIndex.size(), false);
  /// if (result == nullptr) {
  ///    std::cout << "Check failed" << std::endl;
  /// }

  static Operation *matchLine(SmallVector<Operation *> &lineOut,
                              Operation *inputOp, SmallVector<int> &inputIndex,
                              int originLength, bool ifCheckLast = true) {
    if (sizeof...(T2) != inputIndex.size()) {
      return nullptr;
    }
    auto op = llvm::dyn_cast_or_null<T1>(inputOp);
    if (op == nullptr) {
      return nullptr;
    }
    lineOut.push_back(inputOp);

    // Check if the operation is the first one.
    // Check if the operation has only one output.
    // Check if the output of the operation has only one user.
    if (ifCheckLast == true && inputIndex.size() == 0 &&
        originLength != (inputIndex.size() + 1) &&
        (inputOp->getNumResults() != 1 || !inputOp->getResult(0).hasOneUse())) {
      return nullptr;
    }

    // Get next op.
    if (inputIndex.begin() == inputIndex.end())
      return inputOp;
    int order = inputIndex.front();
    if (order >= (int)inputOp->getNumOperands()) {
      return nullptr;
    }
    inputIndex.erase(inputIndex.begin());
    // Order equals ANY_INDEX means this op match is order irrelevant. All
    // operands can possbily be matched.
    if (order == ANY_INDEX) {
      for (int operandIdx = 0; operandIdx < inputOp->getNumOperands();
           operandIdx++) {
        Value opNextValue = inputOp->getOperand(operandIdx);
        Operation *opNext = opNextValue.getDefiningOp();
        auto returnOp =
            UpstreamMatcher<T2...>::matchLine(lineOut, opNext, ifCheckLast);
        if (returnOp != nullptr) {
          return returnOp;
        }
      }
      lineOut.pop_back();
      return nullptr;
    } else {
      Value opNextValue = inputOp->getOperand(order);
      Operation *opNext = opNextValue.getDefiningOp();
      return UpstreamMatcher<T2...>::matchLine(lineOut, opNext, inputIndex,
                                               originLength, ifCheckLast);
    }
  }

  static Operation *matchLine(SmallVector<Operation *> &lineOut,
                              Operation *inputOp, bool ifCheckLast = true) {
    auto op = llvm::dyn_cast_or_null<T1>(inputOp);
    if (op == nullptr) {
      return nullptr;
    }
    lineOut.push_back(inputOp);

    // Check if the operation is the first one.
    // Check if the operation has only one output.
    // Check if the output of the operation has only one user.
    if (sizeof...(T2) == 0) {
      if ((ifCheckLast == true) && ((inputOp->getNumResults() != 1) ||
                                    (!inputOp->getResult(0).hasOneUse()))) {
        return nullptr;
      }
      return inputOp;
    }

    // Get next op.
    for (int operandIdx = 0; operandIdx < inputOp->getNumOperands();
         operandIdx++) {
      Value opNextValue = inputOp->getOperand(operandIdx);
      Operation *opNext = opNextValue.getDefiningOp();
      auto returnOp =
          UpstreamMatcher<T2...>::matchLine(lineOut, opNext, ifCheckLast);
      if (returnOp != nullptr) {
        return returnOp;
      }
    }
    lineOut.pop_back();
    return nullptr;
  }
};

template <> struct UpstreamMatcher<> {
  static Operation *matchLine(SmallVector<Operation *> &lineOut,
                              Operation *inputOp, SmallVector<int> &inputOrder,
                              int originLength, bool ifCheckLast = true) {
    return nullptr;
  }
  static Operation *matchLine(SmallVector<Operation *> &lineOut,
                              Operation *inputOp, bool ifCheckLast = true) {
    return nullptr;
  }
};

Operation *upstreamMatcher(SmallVector<SmallVector<Operation *>> &lineOut,
                           Operation *inputOp, bool ifCheckLast = true);

/// A enum class for representing reduction mode.
enum class ReductionMode {
  SUM,
  MAX,
  UMAX,
  NAN_MAX,
  MIN,
  UMIN,
  NAN_MIN,
  PROD,
  AND,
  OR,
  XOR,
  ARGMAX,
  ARGMIN
};

/// Check whether the reduce op is supported and get the reduction mode
/// if supported.
std::optional<ReductionMode> getReductionMode(triton::ReduceOp op);

/// Check whether the reduce op can convert to argmax/min operation.
std::optional<ReductionMode> matchArgMaxMinPattern(triton::ReduceOp op);

/// Identify the pattern of the reduce operator.
std::optional<ReductionMode> reducePatternRecognition(triton::ReduceOp op);

/// Check whether the reduce operation is constructed with a single
/// statement with type `OpTy`. And the statement has two arguments
/// from the block argument. And the operand of the yield operation
/// is the result of the single statement.
template <typename OpTy, typename ReduceTy>
static bool isSingleStatementReduceOpWithType(ReduceTy op) {
  // Block *block = op.getBlock();
  Block *block = &op.getRegion().front();
  Operation *initFirstPayloadOp =
      triton::linalg_ext::findPayloadOp(block, true);
  Operation *initBackPayloadOp =
      triton::linalg_ext::findPayloadOp(block, false);
  return (isa_and_nonnull<OpTy>(initFirstPayloadOp)) ||
         (isa_and_nonnull<OpTy>(initBackPayloadOp));
}

} // namespace triton
} // namespace mlir
#endif // TRITON_LINALG_DIALECT_LINALGEXT_UTILS_UTILS_H
