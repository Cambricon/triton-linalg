//===- AuxiliaryDialect.cpp - Auxiliary operations imple------- -*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
// This file implementations the auxiliary operations.
//===----------------------------------------------------------------------===//

#include <optional>
#include <stdint.h>
#include <string>
#include <tuple>

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryDialect.h"
#include "llvm/ADT/ilist_iterator.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#define DEBUG_TYPE "aux-dialect"

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOpsDialect.cpp.inc"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::aux;
namespace mlir {
class IRMapping;
class Dialect;
class NamedAttribute;
} // namespace mlir
//===----------------------------------------------------------------------===//
// AuxiliaryDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with auxiliary
/// operations.
struct AuxiliaryInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//
  /// All operations within auxiliary ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// DispatchOp must match ExecutableOp with clone inline.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Regions can always be inlined into functions.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// custom<SymbolAlias>($sym_name, $alias)
//===----------------------------------------------------------------------===//
//  @foo            sym_name: @foo, alias: @foo
//  @foo as("bar")  sym_name: @bar, alias: @foo
ParseResult parseSymbolAlias(OpAsmParser &parser, StringAttr &symName,
                             FlatSymbolRefAttr &alias) {
  if (failed(parser.parseAttribute(alias))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("as"))) {
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(symName)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    symName = StringAttr::get(parser.getContext(), alias.getValue());
  }
  return success();
}

void printSymbolAlias(OpAsmPrinter &p, Operation *op, StringAttr symName,
                      FlatSymbolRefAttr alias) {
  p.printAttributeWithoutType(alias);
  if (symName.getValue() != alias.getValue()) {
    p << " as(\"" << symName.getValue() << "\")";
  }
}

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Dialect/Arith/IR/ArithOps.cpp
//===----------------------------------------------------------------------===//

template <typename... Types> using type_list = std::tuple<Types...> *;

/// Returns a non-null type only if the provided type is one of the allowed
/// types or one of the allowed shaped types of the allowed types. Returns the
/// element type if a valid shaped type is provided.
template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>,
                              type_list<ElementTypes...>) {
  if (llvm::isa<ShapedType>(type) && !llvm::isa<ShapedTypes...>(type))
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!llvm::isa<ElementTypes...>(underlyingType))
    return {};

  return underlyingType;
}

/// Get allowed underlying types for vectors, tensors, and memrefs.
template <typename... ElementTypes>
static Type getTypeIfLikeOrMemRef(Type type) {
  return getUnderlyingType(type,
                           type_list<VectorType, TensorType, MemRefType>(),
                           type_list<ElementTypes...>());
}

/// Return false if both types are ranked tensor with mismatching encoding.
static bool hasSameEncoding(Type typeA, Type typeB) {
  auto rankedTensorA = dyn_cast<RankedTensorType>(typeA);
  auto rankedTensorB = dyn_cast<RankedTensorType>(typeB);
  if (!rankedTensorA || !rankedTensorB)
    return true;
  return rankedTensorA.getEncoding() == rankedTensorB.getEncoding();
}

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Dialect/Arith/IR/ArithOps.cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StoreResourceOp
//===----------------------------------------------------------------------===//
LogicalResult StoreResourceOp::verify() {
  if (!(hasPureTensorSemantics() || hasPureBufferSemantics())) {
    return emitOpError() << "unsupported 'from' and 'to' type.";
  }
  auto from = getFrom();
  auto to = getTo();
  if (isScalar(from) || isa<::mlir::TensorType>(from.getType())) {
    if (from.getType() != to.getType()) {
      return emitOpError()
             << "failed to verify that all of {from, to} have same type";
    }
    return success();
  }
  if (getElementTypeOrSelf(from) != getElementTypeOrSelf(to)) {
    return emitOpError()
           << "failed to verify that all of {from, to} have same element type";
  }

  if (failed(verifyCompatibleShapes(
          mlir::TypeRange{from.getType(), to.getType()}))) {
    return emitOpError()
           << "failed to verify that all of {from, to} have same shape";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//

void ViewOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view_memref");
}

/// Build a ViewOp with all dynamic entries: `staticOffsets`,
/// `staticSizes` and `staticStrides` are automatically filled with
/// source-memref-rank sentinel values that encode dynamic entries.
void ViewOp::build(OpBuilder &b, OperationState &result, MemRefType resultType,
                   Type elementType, Value source, OpFoldResult offset,
                   ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                   StringAttr cacheMode, ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offset, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = cast<LLVM::LLVMPointerType>(source.getType());
  if (!resultType) {
    resultType = MemRefType::get(
        staticSizes, elementType,
        makeStridedLinearLayoutMap(staticStrides, staticOffsets[0],
                                   sourceType.getContext()),
        sourceType.getAddressSpace());
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides), cacheMode);
  result.addAttributes(attrs);
}

void ViewOp::build(OpBuilder &b, OperationState &result, Type elementType,
                   Value source, OpFoldResult offset,
                   ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                   StringAttr cacheMode, ArrayRef<NamedAttribute> attrs) {
  return build(b, result, MemRefType(), elementType, source, offset, sizes,
               strides, cacheMode, attrs);
}

void ViewOp::build(OpBuilder &b, OperationState &result, MemRefType resultType,
                   Type elementType, Value source, int64_t offset,
                   ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                   StringAttr cacheMode, ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(sizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, elementType, source, b.getI64IntegerAttr(offset),
        sizeValues, strideValues, cacheMode, attrs);
}

void ViewOp::build(OpBuilder &b, OperationState &result, MemRefType resultType,
                   Type elementType, Value source, Value offset,
                   ValueRange sizes, ValueRange strides, StringAttr cacheMode,
                   ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, elementType, source, offset, sizeValues,
        strideValues, cacheMode, attrs);
}

void ViewOp::build(OpBuilder &b, OperationState &result, Type elementType,
                   Value source, Value offset, ValueRange sizes,
                   ValueRange strides, StringAttr cacheMode,
                   ArrayRef<NamedAttribute> attrs) {
  build(b, result, MemRefType(), elementType, source, offset, sizes, strides,
        cacheMode, attrs);
}

// TODO: ponder whether we want to allow missing trailing sizes/strides that are
// completed automatically, like we have for subview and extract_slice.
LogicalResult ViewOp::verify() {
  // The source and result memrefs should be in the same memory space.
  auto srcType = llvm::cast<LLVM::LLVMPointerType>(getPtr().getType());
  auto resultType = llvm::cast<MemRefType>(getType());
  if (srcType.getAddressSpace() != resultType.getMemorySpaceAsInt())
    return emitError("different memory spaces specified for source type ")
           << srcType << " and result memref type " << resultType;

  // Match sizes in result memref type and in static_sizes attribute.
  for (const auto &en :
       llvm::enumerate(llvm::zip(resultType.getShape(), getStaticSizes()))) {
    int64_t resultSize = std::get<0>(en.value());
    int64_t expectedSize = std::get<1>(en.value());
    if (!ShapedType::isDynamic(resultSize) &&
        !ShapedType::isDynamic(expectedSize) && resultSize != expectedSize)
      return emitError("expected result type with size = ")
             << expectedSize << " instead of " << resultSize
             << " in dim = " << en.index();
  }

  // Match offset and strides in static_offset and static_strides attributes. If
  // result memref type has no affine map specified, this will assume an
  // identity layout.
  int64_t resultOffset;
  SmallVector<int64_t, 4> resultStrides;
  if (failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
    return emitError("expected result type to have strided layout but found ")
           << resultType;

  // Match offset in result memref type and in static_offsets attribute.
  int64_t expectedOffset = getStaticOffsets().front();
  if (!ShapedType::isDynamic(resultOffset) &&
      !ShapedType::isDynamic(expectedOffset) && resultOffset != expectedOffset)
    return emitError("expected result type with offset = ")
           << resultOffset << " instead of " << expectedOffset;

  // Match strides in result memref type and in static_strides attribute.
  for (const auto &en :
       llvm::enumerate(llvm::zip(resultStrides, getStaticStrides()))) {
    int64_t resultStride = std::get<0>(en.value());
    int64_t expectedStride = std::get<1>(en.value());
    if (!ShapedType::isDynamic(resultStride) &&
        !ShapedType::isDynamic(expectedStride) &&
        resultStride != expectedStride)
      return emitError("expected result type with stride = ")
             << expectedStride << " instead of " << resultStride
             << " in dim = " << en.index();
  }

  // Check cache mode legality.
  auto cacheMode = getCacheMode();
  if (cacheMode.has_value()) {
    llvm::SmallVector<std::string, 2> modes{"cmnormal", "cmtransient"};
    if (llvm::find(modes, cacheMode.value().str()) == modes.end()) {
      return emitError("expected legal cache mode but found ")
             << cacheMode.value();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
void PrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto operand : getDpsInputs()) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (auto operand : getDpsInits()) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), 0, false,
                       SideEffects::DefaultResource::get());
}

LogicalResult PrintOp::verify() {
  if (getNumOperands() > 1) {
    return emitOpError("only accepts 1 input operand atmost!\n");
  }

  if (getFormatAttr()) {
    StringRef fmtValue = getFormatAttr().getValue();
    if (!fmtValue.contains('%')) {
      return success();
    }
    auto totalFmts = fmtValue.count(StringRef("%"));
    auto inValidFmts = fmtValue.count(StringRef("%%"));
    auto validFmtNum = totalFmts - inValidFmts * 2;
    if (validFmtNum != 1) {
      return emitOpError() << "Expected valid format num is 1 but now it is "
                           << validFmtNum << "\n";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ScalarPrintOp
//===----------------------------------------------------------------------===//
void ScalarPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), 0, false,
                       SideEffects::DefaultResource::get());
}

LogicalResult ScalarPrintOp::verify() {
  int64_t numOfOperands = getNumOperands();
  if (numOfOperands > 1)
    return emitOpError("only accepts 1 input operand atmost!\n");

  if (getFormatAttr()) {
    StringRef fmtValue = getFormatAttr().getValue();
    if (!fmtValue.contains('%')) {
      return success();
    }
    auto totalFmts = fmtValue.count(StringRef("%"));
    auto inValidFmts = fmtValue.count(StringRef("%%"));
    auto validFmtNum = totalFmts - inValidFmts * 2;
    if (validFmtNum != numOfOperands) {
      return emitOpError() << "Operands num " << numOfOperands
                           << " need equal to valid fmt num " << validFmtNum
                           << "\n";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BitcastExtOp
//===----------------------------------------------------------------------===//
void BitcastExtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto operand : getDpsInputs()) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (auto operand : getDpsInits()) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand,
                         SideEffects::DefaultResource::get());
  }
}

LogicalResult BitcastExtOp::verify() {
  ShapedType inType = cast<ShapedType>(getIn().getType());
  ShapedType outType = cast<ShapedType>(getOut().getType());
  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();
  if (inRank != outRank) {
    return emitOpError() << "Input and output need to have the same rank.";
  }
  if (inType.getElementTypeBitWidth() == outType.getElementTypeBitWidth()) {
    return emitOpError()
           << "Input and output require different data types and bit widths.";
  }
  // Check continuous except for the lowest dimension.
  for (int64_t dim = 0; dim < inRank - 1; ++dim) {
    if (inShape[dim] != outShape[dim]) {
      return emitOpError() << "Shape mismatch at dimension " << dim
                           << " between in and out";
    }
  }
  // The minimum dimensional dim size for input and output is
  // a multiple relationship.
  int64_t largerDim = std::max(inShape[inRank - 1], outShape[inRank - 1]);
  int64_t smallerDim = std::min(inShape[inRank - 1], outShape[inRank - 1]);
  if (largerDim % smallerDim != 0) {
    return emitOpError()
           << "The lowest dimension of in and out are not multiples.";
  }

  if (auto memType = dyn_cast<MemRefType>(getIn().getType())) {
    auto inBitwidth = memType.getElementTypeBitWidth();
    auto outMemType = dyn_cast<MemRefType>(getOut().getType());
    auto outBitwidth = outMemType.getElementTypeBitWidth();
    bool isLowToHigh = inBitwidth < outBitwidth ? 1 : 0;
    auto rank = memType.getRank();
    auto stride = getStridesAndOffset(memType).first;
    int64_t inLastDim = memType.getDimSize(rank - 1);

    if (isLowToHigh && (stride[rank - 1] != 1))
      return emitOpError()
             << "When the low bit width transitions to the high bit width, "
                "constrain its lowest dimensional continuity";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AuxiliaryDialect
//===----------------------------------------------------------------------===//
void aux::AuxiliaryDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOps.cpp.inc"
      >();
  addInterfaces<AuxiliaryInlinerInterface>();
}

#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/Auxiliary/IR/AuxiliaryOps.cpp.inc"
