//===- LinalgExtDialect.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//
#include <initializer_list>
#include <optional>
#include <stdint.h>
#include <type_traits>
#include <utility>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
// IWYU pragma: begin_keep
// For .inc file only.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/InliningUtils.h"
// IWYU pragma: end_keep
namespace mlir {
class IRMapping;
class Dialect;
class Region;
} // namespace mlir
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::linalg_ext;

//===----------------------------------------------------------------------===//
// LinalgExtDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct LinalgExtInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in Linalg dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {}
};

} // namespace

//===----------------------------------------------------------------------===//
// LinalgExtDialect
//===----------------------------------------------------------------------===//

/// Trait to check if T provides a `regionBuilder` method.
template <typename T, typename... Args>
using has_region_builder = decltype(T::regionBuilder);
template <typename T>
using detect_has_region_builder = llvm::is_detected<has_region_builder, T>;

/// SFINAE helper for single C++ class without a `regionBuilder` method (e.g.
/// an OpInterface).
template <typename OpType, typename = std::enable_if_t<
                               !detect_has_region_builder<OpType>::value>>
void addNamedOpBuilderImpl(
    llvm::StringMap<linalg_ext::LinalgExtDialect::RegionBuilderFunType> &map) {
  // Do nothing.
}

template <typename OpType,
          typename = std::enable_if_t<detect_has_region_builder<OpType>::value>,
          typename = void>
void addNamedOpBuilderImpl(
    llvm::StringMap<linalg_ext::LinalgExtDialect::RegionBuilderFunType> &map) {
  map.insert(std::make_pair(
      OpType::getOperationName(),
      static_cast<linalg_ext::LinalgExtDialect::RegionBuilderFunType>(
          OpType::regionBuilder)));
}

template <typename... OpTypes>
void addNamedOpBuilders(
    llvm::StringMap<linalg_ext::LinalgExtDialect::RegionBuilderFunType> &map) {
  (void)std::initializer_list<int>{0,
                                   (addNamedOpBuilderImpl<OpTypes>(map), 0)...};
}

void mlir::triton::linalg_ext::LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtStructedOps.cpp.inc"
      >();

  // Fill the Linalg-specific OpName to RegionBuilder map.
  addNamedOpBuilders<
#define GET_OP_LIST
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >(namedStructuredOpRegionBuilders);

  addNamedOpBuilders<
#define GET_OP_LIST
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtStructedOps.cpp.inc"
      >(namedStructuredOpRegionBuilders);

  addInterfaces<LinalgExtInlinerInterface>();
}

LogicalResult LinalgExtDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  if (attr.getName() == linalg::LinalgDialect::kMemoizedIndexingMapsAttrName)
    return success();
  return op->emitError() << "attribute '" << attr.getName()
                         << "' not supported by the linalg dialect";
}

#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtEnums.cpp.inc"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOpsDialect.cpp.inc"
