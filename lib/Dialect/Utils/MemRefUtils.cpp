//===- MemRefUtils.cpp - Helpers related to memref Dialect ------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <functional>
#include <numeric>
#include <optional>
#include <stddef.h>
#include <stdint.h>
#include <tuple>

#include "triton-linalg/Dialect/Utils/MemRefUtils.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <mlir/IR/AffineMap.h>

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

/// Try to get the broadcast dimensions from 'srcTy' to 'dstTy', if successful,
/// return the broadcast dimensions, otherwise return failure. The broadcast
/// rules are as follows: 1)The rank of ``dstTy`` must be greater than the rank
/// of ``srcTy``, 2)The size of broadcast dimensions must be equal to the
/// result of substracting between dst rank and src rank, 3)``srcTy`` and
/// ``dstTy`` should have the same static shape except the dimensions in
/// broadcast dimensions.
FailureOr<SmallVector<int64_t>>
mlir::triton::getBroadcastDimensions(MemRefType dstTy, MemRefType srcTy) {
  assert(dstTy.getElementType() == srcTy.getElementType() &&
         "dstTy and srcTy should have the same element type");
  auto srcShapes = srcTy.getShape();
  auto dstShapes = dstTy.getShape();
  int64_t srcRank = srcShapes.size();
  int64_t dstRank = dstShapes.size();
  if (dstRank <= srcRank) {
    return failure();
  }
  SmallVector<int64_t> broadcastDims;

  // Find mapping from src dims to dst dims.
  SmallVector<int64_t> dimMaps;
  int dstId = dstRank - 1;
  for (int srcId = srcRank - 1; srcId >= 0; --srcId) {
    while (dstId >= 0 && ((srcShapes[srcId] != dstShapes[dstId]) ||
                          (ShapedType::isDynamic(dstShapes[dstId]) &&
                           ShapedType::isDynamic(srcShapes[srcId])))) {
      dstId--;
    }
    if (dstId < 0)
      return failure();
    dimMaps.push_back(dstId);
    dstId--;
  }

  if (dimMaps.size() != srcRank) {
    return failure();
  }

  for (auto dim : llvm::seq<int64_t>(0, dstRank)) {
    if (!llvm::is_contained(dimMaps, dim)) {
      broadcastDims.push_back(dim);
    }
  }
  return broadcastDims;
}
