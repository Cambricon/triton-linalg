//===- ShapeUtils.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <iterator>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
using namespace mlir;

bool mlir::triton::isConsecutive(llvm::ArrayRef<int64_t> array) {
  return llvm::all_of(llvm::enumerate(array), [array](auto iter) {
    return iter.index() + array.front() == iter.value();
  });
}

bool mlir::triton::trailingNDimsContiguous(MemRefType type, int64_t n) {
  if (canonicalizeStridedLayout(type).getLayout().isIdentity())
    return true;

  auto memrefShape = type.getShape().take_back(n);
  if (ShapedType::isDynamicShape(memrefShape.drop_front()))
    return false;

  int64_t offset;
  SmallVector<int64_t> stridesFull;
  if (!succeeded(getStridesAndOffset(type, stridesFull, offset)))
    return false;
  auto strides = ArrayRef<int64_t>(stridesFull).take_back(n);

  if (strides.empty())
    return true;

  // Check whether strides match "flattened" dims.
  SmallVector<int64_t> flattenedDims;
  auto dimProduct = 1;
  for (auto dim : llvm::reverse(memrefShape.drop_front(1))) {
    dimProduct *= dim;
    flattenedDims.push_back(dimProduct);
  }

  strides = strides.drop_back(1);
  return llvm::equal(strides, llvm::reverse(flattenedDims));
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
Value mlir::triton::getSlice(OpBuilder &b, Location loc, Value source,
                             ArrayRef<OpFoldResult> offsets,
                             ArrayRef<OpFoldResult> sizes,
                             ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}

bool mlir::triton::isNoTile(OpFoldResult tileSize, OpFoldResult offset,
                            ArrayRef<int64_t> shape, int64_t dim) {
  auto maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return maybeIntTileSize.value() == 0 ||
           maybeIntTileSize.value() == shape[dim];
  }
  auto maybeIntOffset = getConstantIntValue(offset);
  return maybeIntOffset.has_value();
}

OpFoldResult mlir::triton::canonicalizeOpFoldResult(OpFoldResult in) {
  if (in.is<Attribute>())
    return in;
  return getAsOpFoldResult(in.get<Value>());
}

SmallVector<OpFoldResult>
mlir::triton::canonicalizeOpFoldResult(ArrayRef<OpFoldResult> in) {
  return llvm::to_vector(llvm::map_range(in, [](OpFoldResult ofr) {
    return mlir::triton::canonicalizeOpFoldResult(ofr);
  }));
}

Value mlir::triton::getDimValue(OpBuilder &builder, Location loc, Value v,
                                int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult mlir::triton::getDim(OpBuilder &builder, Location loc, Value v,
                                  int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

SmallVector<OpFoldResult>
mlir::triton::getDims(OpBuilder &builder, Location loc, Value shapedTypeValue) {
  SmallVector<OpFoldResult> ret;
  for (auto i : llvm::seq<int64_t>(
           0, cast<ShapedType>(shapedTypeValue.getType()).getRank())) {
    ret.push_back(getDim(builder, loc, shapedTypeValue, i));
  }
  return ret;
}

SmallVector<Value> mlir::triton::getDimsValue(OpBuilder &builder, Location loc,
                                              Value shapedTypeValue) {
  SmallVector<Value> ret;
  for (auto i : llvm::seq<int64_t>(
           0, cast<ShapedType>(shapedTypeValue.getType()).getRank())) {
    ret.push_back(getDimValue(builder, loc, shapedTypeValue, i));
  }
  return ret;
}

SmallVector<Value> mlir::triton::getDynamicDimsValue(OpBuilder &builder,
                                                     Location loc, Value val) {
  SmallVector<Value> dynamicDims;
  auto type = cast<ShapedType>(val.getType());
  for (auto dimIdx : llvm::seq<int64_t>(0, type.getRank())) {
    if (type.isDynamicDim(dimIdx)) {
      dynamicDims.push_back(getDimValue(builder, loc, val, dimIdx));
    }
  }
  return dynamicDims;
}

Value mlir::triton::materializeOpFoldResult(OpBuilder &builder, Location loc,
                                            OpFoldResult opFoldResult) {
  if (auto value = dyn_cast<Value>(opFoldResult))
    return value;
  auto attr = cast<IntegerAttr>(opFoldResult.get<Attribute>());
  return builder.create<arith::ConstantIndexOp>(loc,
                                                attr.getValue().getSExtValue());
}

Value mlir::triton::prependUnitDim(OpBuilder &b, Location loc, Value value) {
  auto valTy = cast<ShapedType>(value.getType());
  int64_t rank = valTy.getRank();
  SmallVector<int64_t> shape(valTy.getShape());
  shape.insert(shape.begin(), 1);

  SmallVector<ReassociationIndices> reassociation;
  if (rank > 0) {
    for (int64_t i = 0; i < rank; ++i)
      reassociation.push_back({i + 1});
    reassociation[0].insert(reassociation[0].begin(), 0);
  }

  return TypeSwitch<ShapedType, Value>(valTy)
      .Case<RankedTensorType>([&](auto) {
        auto expandedType = valTy.cloneWith(shape, valTy.getElementType());
        return b.create<tensor::ExpandShapeOp>(loc, expandedType, value,
                                               reassociation);
      })
      .Case<MemRefType>([&](auto) {
        return b.create<memref::ExpandShapeOp>(loc, shape, value,
                                               reassociation);
      })
      .Default([](auto) -> Value { llvm_unreachable("unsupport value type"); });
}

Value mlir::triton::dropUnitFirstDim(OpBuilder &b, Location loc, Value value) {
  auto valTy = cast<ShapedType>(value.getType());
  int64_t rank = valTy.getRank();
  assert(rank > 0 && valTy.getShape().front() == 1);

  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 1; i < rank; ++i)
    reassociation.push_back({i});
  if (rank > 1)
    reassociation[0].insert(reassociation[0].begin(), 0);

  return TypeSwitch<ShapedType, Value>(valTy)
      .Case<RankedTensorType>([&](auto) {
        return b.create<tensor::CollapseShapeOp>(loc, value, reassociation);
      })
      .Case<MemRefType>([&](auto) {
        return b.create<memref::CollapseShapeOp>(loc, value, reassociation);
      })
      .Default([](auto) -> Value { llvm_unreachable("unsupport value type"); });
}

Value mlir::triton::appendUnitDim(OpBuilder &b, Location loc, Value value) {
  auto valTy = cast<ShapedType>(value.getType());
  int64_t rank = valTy.getRank();
  SmallVector<int64_t> shape(valTy.getShape());
  shape.push_back(1);

  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < rank; ++i)
    reassociation.push_back({i});

  if (!reassociation.empty())
    reassociation.back().push_back(rank);

  return TypeSwitch<ShapedType, Value>(valTy)
      .Case<RankedTensorType>([&](auto) {
        auto expandedType = valTy.cloneWith(shape, valTy.getElementType());
        return b.create<tensor::ExpandShapeOp>(loc, expandedType, value,
                                               reassociation);
      })
      .Case<MemRefType>([&](auto) {
        return b.create<memref::ExpandShapeOp>(loc, shape, value,
                                               reassociation);
      })
      .Default([](auto) -> Value { llvm_unreachable("unsupport value type"); });
}

static bool DetermineLastNDContiguous(MemRefType type, int64_t n,
                                      bool exceptLastDim) {
  int64_t idx = type.getRank();
  for (; idx > 0; idx--) {
    if (mlir::triton::trailingNDimsContiguous(type, idx))
      break;
  }
  return idx >= n + static_cast<int>(exceptLastDim);
}

Value mlir::triton::collapseLastNDimsToOneDim(OpBuilder &b, Location loc,
                                              Value value, int64_t n,
                                              bool exceptLastDim) {
  if (!value || n == 1)
    return value;

  if (isa<MemRefType>(value.getType())) {
    assert(DetermineLastNDContiguous(cast<MemRefType>(value.getType()), n,
                                     exceptLastDim) &&
           "The dimensions that require collapse need to be continuous.");
  }
  auto valueTy = cast<ShapedType>(value.getType());
  auto rank = valueTy.getRank();
  if (exceptLastDim)
    rank -= 1;
  assert(rank >= n && "Dim number to collapse is larger than rank.");

  // When exceptLastDim is false and n == 0, add a unit dim to the last.
  if (n == 0 && !exceptLastDim)
    return appendUnitDim(b, loc, value);

  // Collapse the last n(n > 1) dims to one dim.
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < rank - n; ++i)
    reassociation.push_back({i});
  reassociation.push_back(llvm::to_vector(llvm::seq<int64_t>(rank - n, rank)));
  if (exceptLastDim)
    reassociation.push_back({rank});
  return TypeSwitch<ShapedType, Value>(valueTy)
      .Case<RankedTensorType>([&](auto) {
        return b.create<tensor::CollapseShapeOp>(loc, value, reassociation);
      })
      .Case<MemRefType>([&](auto) {
        return b.create<memref::CollapseShapeOp>(loc, value, reassociation);
      });
}

bool mlir::triton::isScalar(const Value val) {
  return !isa<ShapedType>(val.getType());
}
