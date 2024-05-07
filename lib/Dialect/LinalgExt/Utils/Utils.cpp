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

#include "triton-linalg/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"

using namespace mlir;

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
