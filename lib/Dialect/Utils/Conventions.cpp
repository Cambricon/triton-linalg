//===- Conventions.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include <assert.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton-linalg/Dialect/Utils/Conventions.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::triton;

bool mlir::triton::isLinearMemory(::mlir::ModuleOp op) {
  if (auto attr = op->getAttr(getIsLinearMemoryAttrKey())) {
    assert(attr.dyn_cast<BoolAttr>() && "Invalid linear attribute type");
    return attr.cast<BoolAttr>().getValue();
  }

  // The default value for missing linear attribute is false.
  return false;
}
