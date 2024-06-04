//===- LinalgExtOps.cpp - Imple of the linalg ext operations-----*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LinalgExt operations.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::triton;
using namespace mlir::triton::linalg_ext;

// Helper builders from LinalgOps.cpp.
static void buildGenericRegion(
    OpBuilder &builder, Location loc, Region &region, ValueRange inputs,
    ValueRange outputs,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      blockArgTypes.push_back(getElementTypeOrSelf(v));
      blockArgLocs.push_back(v.getLoc());
    }
  }
  OpBuilder::InsertionGuard guard(builder);
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, loc, bodyBlock->getArguments());
}

template <class Yield = linalg::YieldOp>
static void buildIdentityRegion(OpBuilder &builder, Location loc,
                                Region &region, ValueRange inputs,
                                ValueRange outputs) {
  buildGenericRegion(builder, loc, region, inputs, outputs,
                     [](OpBuilder &b, Location loc, ValueRange args) {
                       b.create<Yield>(loc, args[0]);
                     });
}
// Helper function for getEffect impl from LinalgOps.cpp.
static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, const ValueRange inputOperands,
    ValueRange outputOperands) {
  for (auto operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (auto operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
//===----------------------------------------------------------------------===//
static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    LinalgOp linalgOp) {
  SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
  for (auto [index, operand] : llvm::enumerate(inputOperands)) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(&linalgOp->getOpOperand(index))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
  }
  unsigned inputOperandSize = inputOperands.size();

  for (auto [index, operand] : llvm::enumerate(linalgOp.getDpsInits())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(
            &linalgOp->getOpOperand(index + inputOperandSize))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parse and print functions from LinalgOps.cpp
//===----------------------------------------------------------------------===//
/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    result.addAttribute("operand_segment_sizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputTypes.size()),
                             static_cast<int32_t>(outputTypes.size())}));
  }
  return success();
}

static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

//===----------------------------------------------------------------------===//
// Helper functions for named Linalg ops defined in ods-gen from LinalgOps.cpp.
//===----------------------------------------------------------------------===//

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }
  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);
}

/// Creates a structured operation given `inputs`, `outputs`, and `attributes`.
/// The result types are derived automatically if `resultTensorTypes` is none.
/// The body of the operation is filled using `regionBuilder`. All ods-gen
/// created structured operations use the method to implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              std::optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder,
                              bool addOperandSegmentSizes = true) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    llvm::copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
                  [](Type type) { return type.isa<RankedTensorType>(); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  if (addOperandSegmentSizes)
    state.addAttribute(
        "operand_segment_sizes",
        b.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                                static_cast<int32_t>(outputs.size())}));
  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

//===----------------------------------------------------------------------===//
// Specific parsing and printing for named structured ops created by ods-gen
// from LinalgOps.cpp.
//===----------------------------------------------------------------------===//
static ParseResult parseNamedStructuredOpRegion(
    OpAsmParser &parser, Region &region, unsigned numRegionArgs,
    TypeRange inputTypes, TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    RegionBuilderFn regionBuilder) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, region, inputTypes, outputTypes, attrs,
                         regionBuilder);
  return success();
}

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

static void printNamedStructuredOp(OpAsmPrinter &p, Operation *op,
                                   ValueRange inputs, ValueRange outputs) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operand_segment_sizes",
                       // See generated code in
                       // LinalgNamedStructuredOps.yamlgen.cpp.inc
                       "linalg.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, inputs, outputs);

  // Results printing.
  printNamedStructuredOpResults(p, op->getResultTypes());

  // Region is elided.
}

//===----------------------------------------------------------------------===//
// Region builder helper from LinalgOps.cpp.
// TODO: Move this to a utility library.
// The public methods on this class are referenced directly from generated code.
// Helper build the unary, binary, and type conversion functions defined by the
// DSL. See mlir-linalg-ods-yaml-gen.cpp for the code that uses this class.
//
// Implementations of the math functions must be polymorphic over numeric types,
// internally performing necessary casts. If the function application makes no
// sense, then the only recourse is to assert and return nullptr. This can be
// extended later if it becomes possible to fail construction of the region. The
// invariant should be enforced at a higher level.
//
// TODO: These helpers are currently type polymorphic over the class of integer
// and floating point types, but they will not internally cast within bit
// widths of a class (mixed precision such as i8->i32) or across classes
// (i.e. mixed float and integer). Many such combinations are ambiguous or need
// to be handled with care and work is being considered to extend the op
// language to make such cases explicit. In the mean-time, violating this will
// fail verification, which is deemed acceptable.
//===----------------------------------------------------------------------===//

namespace {

class RegionBuilderHelper {
public:
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Build the unary functions defined by OpDSL.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg) {
    if (!isFloatingPoint(arg))
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (unaryFn) {
    case UnaryFn::exp:
      return builder.create<math::ExpOp>(arg.getLoc(), arg);
    case UnaryFn::log:
      return builder.create<math::LogOp>(arg.getLoc(), arg);
    case UnaryFn::abs:
      return builder.create<math::AbsFOp>(arg.getLoc(), arg);
    case UnaryFn::ceil:
      return builder.create<math::CeilOp>(arg.getLoc(), arg);
    case UnaryFn::floor:
      return builder.create<math::FloorOp>(arg.getLoc(), arg);
    case UnaryFn::negf:
      return builder.create<arith::NegFOp>(arg.getLoc(), arg);
    }
    llvm_unreachable("unsupported unary function");
  }

  // Build the binary functions defined by OpDSL.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                   arg1.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (binaryFn) {
    case BinaryFn::add:
      if (allComplex)
        return builder.create<complex::AddOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::AddFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        return builder.create<arith::OrIOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::AddIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::sub:
      if (allComplex)
        return builder.create<complex::SubOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::SubFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        llvm_unreachable("unsupported operation: sub with bools");
      return builder.create<arith::SubIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::mul:
      if (allComplex)
        return builder.create<complex::MulOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::MulFOp>(arg0.getLoc(), arg0, arg1);
      if (allBool)
        return builder.create<arith::AndIOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MulIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaximumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_signed:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinimumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinSIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::max_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MaximumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MaxUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::min_unsigned:
      assert(!allComplex);
      if (allFloatingPoint)
        return builder.create<arith::MinimumFOp>(arg0.getLoc(), arg0, arg1);
      return builder.create<arith::MinUIOp>(arg0.getLoc(), arg0, arg1);
    case BinaryFn::div:
    case BinaryFn::div_unsigned:
      llvm_unreachable("unsupported binary function");
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand) {
    switch (typeFn) {
    case TypeFn::cast_signed:
      return cast(toType, operand, false);
    case TypeFn::cast_unsigned:
      return cast(toType, operand, true);
    }
    llvm_unreachable("unsupported type conversion function");
  }

  void yieldOutputs(ValueRange values) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    builder.create<YieldOp>(loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    auto typedAttr = valueAttr.dyn_cast<TypedAttr>();
    return builder.create<arith::ConstantOp>(loc, typedAttr.getType(),
                                             typedAttr);
  }

  Value index(int64_t dim) {
    OpBuilder builder = getBuilder();
    return builder.create<IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(context, width);
  }

  Type getFloat32Type() { return Float32Type::get(context); }
  Type getFloat64Type() { return Float64Type::get(context); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();

    if (operand.getType() == toType)
      return operand;
    if (auto toIntType = toType.dyn_cast<IntegerType>()) {
      // If operand is floating point, cast directly to the int type.
      if (operand.getType().isa<FloatType>()) {
        if (isUnsignedCast)
          return builder.create<arith::FPToUIOp>(loc, toType, operand);
        return builder.create<arith::FPToSIOp>(loc, toType, operand);
      }
      // Cast index operands directly to the int type.
      if (operand.getType().isIndex())
        return builder.create<arith::IndexCastOp>(loc, toType, operand);
      if (auto fromIntType = operand.getType().dyn_cast<IntegerType>()) {
        // Either extend or truncate.
        if (toIntType.getWidth() > fromIntType.getWidth()) {
          if (isUnsignedCast)
            return builder.create<arith::ExtUIOp>(loc, toType, operand);
          return builder.create<arith::ExtSIOp>(loc, toType, operand);
        }
        if (toIntType.getWidth() < fromIntType.getWidth())
          return builder.create<arith::TruncIOp>(loc, toType, operand);
      }
    } else if (auto toFloatType = toType.dyn_cast<FloatType>()) {
      // If operand is integer, cast directly to the float type.
      // Note that it is unclear how to cast from BF16<->FP16.
      if (operand.getType().isa<IntegerType>()) {
        if (isUnsignedCast)
          return builder.create<arith::UIToFPOp>(loc, toFloatType, operand);
        return builder.create<arith::SIToFPOp>(loc, toFloatType, operand);
      }
      if (auto fromFloatType = operand.getType().dyn_cast<FloatType>()) {
        if (toFloatType.getWidth() > fromFloatType.getWidth())
          return builder.create<arith::ExtFOp>(loc, toFloatType, operand);
        if (toFloatType.getWidth() < fromFloatType.getWidth())
          return builder.create<arith::TruncFOp>(loc, toFloatType, operand);
      }
    }

    emitWarning(operand.getLoc()) << "could not cast operand of type "
                                  << operand.getType() << " to " << toType;
    return operand;
  }

  bool isComplex(Value value) { return value.getType().isa<ComplexType>(); }
  bool isFloatingPoint(Value value) { return value.getType().isa<FloatType>(); }
  bool isInteger(Value value) { return value.getType().isa<IntegerType>(); }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }

  MLIRContext *context;
  Block &block;
};

} // namespace

//===----------------------------------------------------------------------===//
// LibdeviceCallOp
//===----------------------------------------------------------------------===//
void LibdeviceCallOp::build(::mlir::OpBuilder &builder,
                            ::mlir::OperationState &result, ValueRange inputs,
                            Value init, StringAttr symbol,
                            ArrayRef<NamedAttribute> attributes) {
  result.addOperands(inputs);
  result.addOperands(init);
  result.addAttribute(getSymbolAttrName(result.name), symbol);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (initType.isa<RankedTensorType>())
    result.addTypes(initType);
}

void LibdeviceCallOp::build(::mlir::OpBuilder &builder,
                            ::mlir::OperationState &result, ValueRange inputs,
                            Value init, StringRef symbol,
                            ArrayRef<NamedAttribute> attributes) {
  build(builder, result, inputs, init,
        StringAttr::get(builder.getContext(), symbol), attributes);
}

LogicalResult LibdeviceCallOp::verify() { return success(); }

LogicalResult ScalarLibdeviceCallOp::verify() {
  // The inputs of ScalarLibdeviceCallOp should be scalar type.
  for (auto v : getInputs()) {
    if (v.getType().isa<ShapedType>())
      return emitOpError() << "expects all input types are scalar type.";
  }
  // The result type should be scalar type.
  if (getResult().getType().isa<ShapedType>())
    return emitOpError() << "expects the result type is scalar type.";
  return success();
}

//===----------------------------------------------------------------------===//
// Implementation of BatchConv2DNhwcFhwcOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType>
BatchConv2DNhwcFhwcOp::getIteratorTypesArray() {
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction,
      utils::IteratorType::reduction, utils::IteratorType::reduction};
  return iteratorTypes;
}

static SmallVector<AffineExpr>
getBatchConv2DSymbolBindings(BatchConv2DNhwcFhwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

  int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
  exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

  int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
  exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

  int64_t cst7 = self.getStrides().getValues<int64_t>()[1];
  exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

  int64_t cst9 = self.getDilations().getValues<int64_t>()[1];
  exprs.push_back(getAffineConstantExpr(cst9, context));

  exprs.push_back(getAffineSymbolExpr(10, context));
  exprs.push_back(getAffineSymbolExpr(11, context));
  return exprs;
}

ArrayAttr BatchConv2DNhwcFhwcOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getBatchConv2DSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute(
                     "affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, "
                     "s3, s4, s5, s6, s7, s8, s9, s10, s11] -> (d0, d1, d2 * "
                     "s3 + d5 * s5, d3 * s7 + d6 * s9, d7)>",
                     context)
                     .cast<AffineMapAttr>()
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(
      mlir::parseAttribute(
          "affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, "
          "s6, s7, s8, s9, s10, s11] -> (d0, d4, d5, d6, d7)>",
          context)
          .cast<AffineMapAttr>()
          .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(
      mlir::parseAttribute(
          "affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, "
          "s6, s7, s8, s9, s10, s11] -> (d0, d1, d2, d3, d4)>",
          context)
          .cast<AffineMapAttr>()
          .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchConv2DNhwcFhwcOp::getNumRegionArgs() { return 3; }

bool BatchConv2DNhwcFhwcOp::hasDynamicIndexingMaps() { return true; }

LogicalResult BatchConv2DNhwcFhwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
    if (!attr.getType().getElementType().isInteger(64))
      return op->emitError(
          "incorrect element type for index attribute 'strides'");
    if (attr.getType().getShape() != ArrayRef<int64_t>{2})
      return op->emitError("incorrect shape for index attribute 'strides'");
  }
  if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
    if (!attr.getType().getElementType().isInteger(64))
      return op->emitError(
          "incorrect element type for index attribute 'dilations'");
    if (attr.getType().getShape() != ArrayRef<int64_t>{2})
      return op->emitError("incorrect shape for index attribute 'dilations'");
  }
  return success();
}

void BatchConv2DNhwcFhwcOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                          ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchConv2DNhwcFhwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(0));
  Value value2 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(),
                         block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 =
      helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchConv2DNhwcFhwcOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();
  // Add result types.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parser.parseOptionalArrowTypeList(outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);
  // Add region
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, getNumRegionArgs(),
                                   inputTypes, outputTypes,
                                   result.attributes.getAttrs(), regionBuilder))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void BatchConv2DNhwcFhwcOp::print(OpAsmPrinter &p) {
  auto *op = getOperation();
  // Attrs printing.
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operand_segment_sizes",
                                           "linalg.memoized_indexing_maps"});
  // Ins and outs printing.
  printCommonStructuredOpParts(p, getInputs(), getOutputs());
  // Results printing.
  p.printOptionalArrowTypeList(op->getResultTypes());
}

LogicalResult BatchConv2DNhwcFhwcOp::fold(FoldAdaptor,
                                          SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void BatchConv2DNhwcFhwcOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}

//===----------------------------------------------------------------------===//
// Implementation of MakeRangeOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MakeRangeOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr MakeRangeOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap =
      AffineMap::getMultiDimIdentityMap(getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned MakeRangeOp::getNumRegionArgs() { return 3; }

void MakeRangeOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                ArrayRef<NamedAttribute> attrs) {
  assert(block.getNumArguments() == 3 &&
         "MakeRangeOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  Value zero = helper.index(0);
  Value value0 =
      helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), zero);
  Value value1 =
      helper.buildBinaryFn(BinaryFn::add, value0, block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed,
                                    block.getArgument(2).getType(), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult MakeRangeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();
  // Add result types.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parser.parseOptionalArrowTypeList(outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);
  // Add region.
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, getNumRegionArgs(),
                                   inputTypes, outputTypes,
                                   result.attributes.getAttrs(), regionBuilder))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void MakeRangeOp::print(OpAsmPrinter &p) {
  auto *op = getOperation();
  // Attrs printing.
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operand_segment_sizes",
                                           "linalg.memoized_indexing_maps"});
  // Ins and outs printing.
  printCommonStructuredOpParts(p, getInputs(), getOutputs());
  // Results printing.
  p.printOptionalArrowTypeList(op->getResultTypes());
}

LogicalResult MakeRangeOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void MakeRangeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult MakeRangeOp::verify() {
  ShapedType resultType = getOutputOperandType();

  int64_t rank = resultType.getRank();
  if (rank != 1) {
    return emitOpError() << "output rank must be 1 ";
  }
  if (!resultType.getElementType().isInteger(32)) {
    return emitOpError() << "result element type must be i32";
  }

  if (auto start = getStart().getDefiningOp<arith::ConstantOp>()) {
    if (auto end = getEnd().getDefiningOp<arith::ConstantOp>()) {
      int64_t startConstantInt = start.getValue().cast<IntegerAttr>().getInt();
      int64_t endConstantInt = end.getValue().cast<IntegerAttr>().getInt();
      if (endConstantInt <= startConstantInt)
        return emitOpError()
               << "input argument end must greater than input arguments start "
                  "in make range operation";
      SmallVector<int64_t> inputShape = {endConstantInt - startConstantInt};
      auto outputShape = resultType.getShape();
      if (inputShape[0] != outputShape[0]) {
        return emitOpError() << "output shape mismatch";
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Im2ColOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Im2ColOp::getIteratorTypesArray() {
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel};
  return iteratorTypes;
}

static SmallVector<AffineExpr> getSymbolBindings(Im2ColOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

  int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
  exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

  int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
  exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr Im2ColOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(
      mlir::parseAttribute(
          "affine_map<(d0, d1, d2, d4, d5, d6)[s0, s1, s2, s3, s5, s6, s7, "
          "s9] -> (d0, d1 * s2 + d4, d2 * s6 + d5, d6)>",
          context)
          .cast<AffineMapAttr>()
          .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute(
                     "affine_map<(d0, d1, d2, d4, d5, d6)[s0, s1, s2, s3, "
                     "s5, s6, s7, s9] -> (d0, d1, d2, d4, d5, d6)>",
                     context)
                     .cast<AffineMapAttr>()
                     .getValue());
  maps.back() = simplifyAffineMap(
      maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Im2ColOp::getNumRegionArgs() { return 2; }

std::string Im2ColOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Im2ColOp::hasDynamicIndexingMaps() { return true; }

LogicalResult Im2ColOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();

  if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
    if (!attr.getType().getElementType().isInteger(64))
      return op->emitError(
          "incorrect element type for index attribute 'strides'");
    if (attr.getType().getShape() != ArrayRef<int64_t>{2})
      return op->emitError("incorrect shape for index attribute 'strides'");
  }

  return success();
}

void Im2ColOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                             ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "Im2ColOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;

  Value value1 =
      helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(1).getType(),
                         block.getArgument(0));
  yields.push_back(value1);
  helper.yieldOutputs(yields);
}

ParseResult Im2ColOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result, Im2ColOp::getNumRegionArgs(),
                                  Im2ColOp::getRegionBuilder());
}

void Im2ColOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Im2ColOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void Im2ColOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

//===----------------------------------------------------------------------===//
// Implementation of ScatterOp
//===----------------------------------------------------------------------===//
/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate. b) At least one dimension is out of bound (`dimPos` is >= 0 and <
/// rank). c) the number of elements in `dimsPos` is greater than `rank`.
static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  // early exit.
  if (dimsPos.size() > rank)
    return true;
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos)
    uniqued.insert(dim);
  if (dimsPos.size() != uniqued.size())
    return true;
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

static bool isLegalIndiceType(ShapedType indiceType) {
  auto eleTy = indiceType.getElementType();
  return eleTy.isInteger(8) || eleTy.isInteger(16) || eleTy.isInteger(32) ||
         eleTy.isInteger(64);
}

static bool checkDimensionsMatch(ShapedType t1, ShapedType t2,
                                 ArrayRef<unsigned> dims) {
  auto lhsShape = t1.getShape();
  auto rhsShape = t2.getShape();
  return llvm::all_of(dims,
                      [&](auto dim) { return lhsShape[dim] == rhsShape[dim]; });
};

void ScatterOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs, Value init,
    ArrayRef<int64_t> dimensionMap, bool rangedData,
    bool overlapWindow, bool signedIndice,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, init,
        DenseI64ArrayAttr::get(builder.getContext(), dimensionMap), rangedData,
        overlapWindow, signedIndice);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (initType.isa<RankedTensorType>())
    result.addTypes(initType);

  if (bodyBuild) {
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       {inputs.front()}, {init}, bodyBuild);
  } else {
    // default region.
    buildIdentityRegion<linalg_ext::ExtYieldOp>(builder, result.location,
                                                *result.regions.front(),
                                                {inputs.front()}, {init});
  }
}

LogicalResult ScatterOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 2 && getNumDpsInputs() != 3) {
    return op->emitOpError("expected two or three input operands");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }

  auto indiceType = getIndiceType();
  if (!isLegalIndiceType(indiceType)) {
    return op->emitOpError(
        "expected indices to be of rank 2 of i8/i16/i32/i64 element type");
  }

  auto indexDepth = getIndexDepth();
  if (indexDepth == ShapedType::kDynamic) {
    return op->emitOpError("expected index depth is static");
  }
  ArrayRef<int64_t> dimMap = getDimensionMap();
  if (dimMap.size() != indexDepth) {
    return op->emitOpError("invalid number of dimension map entries ");
  }

  auto initType = getInitType();
  if (initType.getRank() < 1) {
    return op->emitOpError("expected init value to be at least rank 1");
  }
  if (isInvalid(dimMap, initType.getRank()))
    return op->emitOpError("dimension map is invalid");
  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto updateType = getUpdateType();
  if (updateType.getRank() < 2) {
    return op->emitOpError("expected update value to be at least rank 2");
  }

  auto batchNum = getBatchDimNum();
  auto batchNumDims = llvm::to_vector(llvm::seq<unsigned>(0, batchNum));
  if (!checkDimensionsMatch(indiceType, updateType, batchNumDims)) {
    return op->emitOpError(
        "mismatch in shape of indices and update value at batch dims");
  }
  if (updateType.getRank() - batchNum != initType.getRank()) {
    return op->emitOpError(
        "update value rank mismatch the rank of the init value");
  }
  if (mask()) {
    auto maskType = getMaskType();
    if (maskType.getRank() != batchNum ||
        !maskType.getElementType().isInteger(1)) {
      return op->emitOpError(
          "expected mask to be of i1 element type and batch matched init");
    }
    if (!checkDimensionsMatch(maskType, updateType, batchNumDims)) {
      return op->emitOpError(
          "mismatch in shape of mask and update value at batch dims");
    }
  }
  // Check that the update indices do not exceed the update length.
  for (auto idx : llvm::seq<unsigned>(0, initType.getRank())) {
    if (!initType.isDynamicDim(idx) &&
        updateType.getDimSize(idx + batchNum) > initType.getDimSize(idx)) {
      return op->emitOpError("indexed shape of update value dim#")
             << idx + batchNum << " exceeds init value at dim#" << idx << " "
             << updateType.getDimSize(idx + batchNum) << " .vs. "
             << initType.getDimSize(idx);
    }
  }
  // A valid region is a region has 2 scalar inputs with same int/float type
  // and returns a single scalar with same type.
  Region &region = this->getRegion();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return op->emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op->emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != initType.getElementType()) {
    return op->emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of init value "
           << initType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op->emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<linalg_ext::ExtYieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

LogicalResult ScatterOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void ScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//
void ScanOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  for (Value v : getRegionInputArgs())
    setNameFn(v, "in");
}

void ScanOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  if (!getResults().empty())
    setNameFn(getResults().front(), "scanned");
}

void ScanOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange inits, ArrayRef<int64_t> dimension,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, inits, dimension);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  for (Value init : inits) {
    Type initType = init.getType();
    if (initType.isa<RankedTensorType>())
      result.addTypes(initType);
  }

  if (bodyBuild)
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       inputs, inits, bodyBuild);
}

void ScanOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult ScanOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult ScanOp::verify() {
  auto numInputs = getNumDpsInputs();
  if ((getNumDpsInits() % 2) != 0) {
    return emitOpError() << "expects outputs paired with inits. "
                         << "but sum of outputs and inits are "
                         << getNumDpsInits() << "."
                         << "num inputs is " << getNumDpsInputs() << ".";
  }

  if (getInits()[0].getType().cast<ShapedType>().getShape() !=
      getInputs()[0].getType().cast<ShapedType>().getShape()) {
    return emitOpError() << "expects inputs and outputs have the same shapes. "
                            "Shape at input-index 0 is not equal to"
                            " the shape at output-index 0.";
  }

  if (getDimensions().size() != 1) {
    return emitOpError() << "only support single dimension.";
  }
  int64_t dimension = getDimensions()[0];

  auto inputType = getInputs()[0].getType().cast<ShapedType>();
  if (dimension < 0 || dimension >= inputType.getRank()) {
    return emitOpError() << "dimension for scan should be in the range [0, "
                         << inputType.getRank() - 1 << "].";
  }

  SmallVector<int64_t> expectedInitShape;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    if (i != dimension)
      expectedInitShape.push_back(inputType.getShape()[i]);
  }

  for (int64_t i = 1; i < numInputs; ++i) {
    if (getInputs()[i].getType().cast<ShapedType>().getShape() !=
        getInputs()[0].getType().cast<ShapedType>().getShape()) {
      return emitOpError() << "expects all inputs have the same shapes. "
                              "Shape at input-index "
                           << i
                           << " is not equal to the shape at input-index 0.";
    }
  }

  auto numOutputs = getNumDpsInits() / 2;
  for (int64_t i = 1; i < numOutputs; ++i) {
    if (getInits()[i].getType().cast<ShapedType>().getShape() !=
        getInits()[0].getType().cast<ShapedType>().getShape()) {
      return emitOpError() << "expects all outputs have the same shapes. "
                              "Shape at output-index "
                           << i
                           << " is not equal to the shape at output-index 0.";
    }
    if (getInits()[i + numOutputs].getType().cast<ShapedType>().getShape() !=
        getInits()[numOutputs].getType().cast<ShapedType>().getShape()) {
      return emitOpError() << "expects all inits have the same shapes. "
                              "Shape at init-index "
                           << i + numOutputs
                           << " is not equal to the shape at init-index "
                           << numOutputs << ".";
    }
  }

  if (expectedInitShape !=
      getInits()[numInputs].getType().cast<ShapedType>().getShape()) {
    return emitOpError() << "inits shape is not equal to the expected shape "
                         << expectedInitShape << ".";
  }

  Block *block = getBody();
  if (block->getNumArguments() != this->getNumOperands())
    return emitOpError()
           << "mismatching number of operands and block arguments";

  // Check that the first block arguments match the element type of the inputs.
  for (auto [input, bbArg] : llvm::zip(getInputs(), block->getArguments())) {
    Type inputElementType = input.getType().cast<ShapedType>().getElementType();
    if (inputElementType != bbArg.getType())
      return emitOpError()
             << "input element type " << inputElementType
             << " does not match corresponding block argument type "
             << bbArg.getType();
  }

  // Check that the last block arguments match the element type of the outputs.
  for (auto [output, bbArg] : llvm::zip(
           getDpsInits(), block->getArguments().take_back(getNumDpsInits()))) {
    Type outputElementType =
        output.getType().cast<ShapedType>().getElementType();
    if (outputElementType != bbArg.getType())
      return emitOpError()
             << "output element type " << outputElementType
             << " does not match corresponding block argument type "
             << bbArg.getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Implementation of GatherOp
//===----------------------------------------------------------------------===//
void GatherOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs, Value init,
    ArrayRef<int64_t> dimensionMap, bool rangedData, bool signedIndice,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, init,
        DenseI64ArrayAttr::get(builder.getContext(), dimensionMap), rangedData, signedIndice);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (initType.isa<RankedTensorType>())
    result.addTypes(initType);

  if (bodyBuild) {
    buildGenericRegion(builder, result.location, *result.regions.front(),
                       {inputs.front()}, {init}, bodyBuild);
  } else {
    // default region.
    buildIdentityRegion<linalg_ext::ExtYieldOp>(builder, result.location,
                                                *result.regions.front(),
                                                {inputs.front()}, {init});
  }
}

LogicalResult GatherOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 2 && getNumDpsInputs() != 3) {
    return op->emitOpError("expected two or three input operands");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }

  auto indiceType = getIndiceType();
  if (!isLegalIndiceType(indiceType)) {
    return op->emitOpError(
        "expected indices to be of rank 2 of i8/i16/i32/i64 element type");
  }

  auto indexDepth = getIndexDepth();
  if (indexDepth == ShapedType::kDynamic) {
    return op->emitOpError("expected index depth is static");
  }
  ArrayRef<int64_t> dimMap = getDimensionMap();
  if (dimMap.size() != indexDepth) {
    return op->emitOpError("invalid number of dimension map entries ");
  }

  auto inputType = getInputType();
  if (inputType.getRank() < 1) {
    return op->emitOpError("expected input value to be at least rank 1");
  }
  if (isInvalid(dimMap, inputType.getRank()))
    return op->emitOpError("dimension map is invalid");
  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto initType = getInitType();
  if (initType.getRank() < 2) {
    return op->emitOpError("expected init value to be at least rank 2");
  }
  auto batchNum = getBatchDimNum();
  auto batchNumDims = llvm::to_vector(llvm::seq<unsigned>(0, batchNum));
  if (!checkDimensionsMatch(indiceType, initType, batchNumDims)) {
    return op->emitOpError(
        "mismatch in shape of indices and init value at batch dims");
  }
  if (initType.getRank() - batchNum != inputType.getRank()) {
    return op->emitOpError(
        "init value rank exceeds the rank of the input value");
  }
  if (mask()) {
    auto maskType = getMaskType();
    if (maskType.getRank() != batchNum ||
        !maskType.getElementType().isInteger(1)) {
      return op->emitOpError(
          "expected mask to be of i1 element type and batch matched init");
    }
    if (!checkDimensionsMatch(maskType, initType, batchNumDims)) {
      return op->emitOpError(
          "mismatch in shape of mask and init value at batch dims");
    }
  }
  // Check that the indices do not exceed the input length.
  for (auto idx : llvm::seq<unsigned>(0, inputType.getRank())) {
    if (!inputType.isDynamicDim(idx) &&
        initType.getDimSize(idx + batchNum) > inputType.getDimSize(idx)) {
      return op->emitOpError("indexed shape of init value dim#")
             << idx + batchNum << " exceeds input value at dim#" << idx << " "
             << initType.getDimSize(idx + batchNum) << " .vs. "
             << inputType.getDimSize(idx);
    }
  }
  // A valid region is a region has 2 scalar inputs with same int/float type
  // and returns a single scalar with same type.
  Region &region = this->getRegion();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return op->emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != initType.getElementType()) {
    return op->emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of init value "
           << initType.getElementType();
  }
  if (arg1Type != inputType.getElementType()) {
    return op->emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of input value "
           << inputType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op->emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<linalg_ext::ExtYieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

LogicalResult GatherOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

//===----------------------------------------------------------------------===//
// Implementation of GatherAtomicRMWOp
//===----------------------------------------------------------------------===//
void GatherAtomicRMWOp::build(OpBuilder &builder, OperationState &result,
                              ValueRange inputs, ValueRange inits,
                              AtomicType atomicType, MemoryOrder memory_order,
                              ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, inits, atomicType, memory_order);
  result.addAttributes(attributes);

  // Add output types for output arguments.
  for (auto i : inits) {
    result.addTypes(i.getType());
  }
}

LogicalResult GatherAtomicRMWOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 2 && getNumDpsInputs() != 3) {
    return op->emitOpError("expected two or three input operands");
  }
  auto indiceType = getIndiceType();
  auto srcType = getSrcType();
  if (indiceType.getRank() != 2)
    return op->emitOpError("expected indice with rank 2");
  if (indiceType.getDimSize(1) != srcType.getRank()) {
    return op->emitOpError("indice dim 1 size should be equal to src rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Implementation of AtomicCASOp
//===----------------------------------------------------------------------===//
LogicalResult AtomicCASOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void AtomicCASOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!hasPureBufferSemantics()) {
    effects.emplace_back(MemoryEffects::Read::get(), input(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), input(),
                         SideEffects::DefaultResource::get());
  }
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

//===----------------------------------------------------------------------===//
// Implementation of GatherAtomicCASOp
//===----------------------------------------------------------------------===//
LogicalResult GatherAtomicCASOp::fold(FoldAdaptor,
                                      SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void GatherAtomicCASOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!hasPureBufferSemantics()) {
    effects.emplace_back(MemoryEffects::Read::get(), input(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), input(),
                         SideEffects::DefaultResource::get());
  }
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult GatherAtomicRMWOp::fold(FoldAdaptor,
                                      SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void GatherAtomicRMWOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto *operand : getDpsInputOperands()) {
    if (!operand->get().getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), src(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), src(),
                       SideEffects::DefaultResource::get());
  if (window().getType().isa<MemRefType>()) {
    effects.emplace_back(MemoryEffects::Write::get(), window(),
                         SideEffects::DefaultResource::get());
  }
}
//===----------------------------------------------------------------------===//
// Implementation of PadOp
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Dialect/Tensor/IR/TensorOps.cpp
//===----------------------------------------------------------------------===//
namespace {
/// Fold chains of tensor::ExtractSliceOp, linalg_ext::PadOp pairs that pad
/// different dimensions. The pattern applies if the following preconditions
/// hold:
///   1) the tensor::ExtractSliceOps are not rank-reducing,
///   2) the tensor::ExtractSliceOps have only unit-strides,
///   3) the linalg_ext::PadOps perform only high-padding,
///   4) the linalg_ext::PadOps have the same constant padding value,
///   5) the linalg_ext::PadOps do not have common padding dimensions,
///   6) one tensor::ExtractSliceOp, linalg_ext::PadOp pair has zero-padding and
///      zero-offset for every dimension.
///   7) the tensor::ExtractSliceOp sizes match the source tensor sizes for the
///      padded source dimensions.
///
/// Example:
///
/// ```mlir
///   %0 = tensor.extract_slice %input[16, 0] [%sz0, 64] [1, 1]
///       : tensor<64x64xf32> to tensor<?x64xf32>
///   %1 = linalg_ext.pad %0 low[0, 0] high[%pw0, 0] { ...
///     } : tensor<?x64xf32> to tensor<8x64xf32>
///   %2 = tensor.extract_slice %1[0, 4] [8, %sz1] [1, 1]
///        : tensor<8x64xf32> to tensor<8x?xf32>
///   %res = linalg_ext.pad %2 nofold low[0, 0] high[0, %pw1] { ...
///     } : tensor<8x?xf32> to tensor<8x4xf32>
/// ```
///
/// folds into:
///
/// ```mlir
///   %0 = tensor.extract_slice %input[16, 4] [%sz0, %sz1] [1, 1]
///        : tensor<64x64xf32> to tensor<?x?xf32>
///   %res = linalg_ext.pad %0 nofold low[0, 0] high[%pw0, %pw1] { ...
///     } : tensor<?x?xf32> to tensor<8x4xf32>
/// ```
struct FoldOrthogonalPaddings : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto innerSliceOp = padOp.input().getDefiningOp<tensor::ExtractSliceOp>();
    if (!innerSliceOp)
      return failure();
    auto outerPadOp = innerSliceOp.getSource().getDefiningOp<PadOp>();
    if (!outerPadOp)
      return failure();
    auto outerSliceOp =
        outerPadOp.input().getDefiningOp<tensor::ExtractSliceOp>();
    if (!outerSliceOp)
      return failure();

    // 1) Fail if the chain is rank-reducing.
    int64_t rank = padOp.getInputType().getRank();
    if (outerSliceOp.getSourceType().getRank() != rank) {
      return rewriter.notifyMatchFailure(padOp,
                                         "cannot fold rank-reducing chain");
    }

    // 2) Fail if the tensor::ExtractSliceOps have non-unit strides.
    if (!innerSliceOp.hasUnitStride() || !outerSliceOp.hasUnitStride()) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold non-unit stride ExtractSliceOps");
    }

    // 3) Fail if the linalg_ext::PadOps have non-zero low padding.
    if (!padOp.hasZeroLowPad() || !outerPadOp.hasZeroLowPad()) {
      return rewriter.notifyMatchFailure(padOp,
                                         "cannot fold PadOps with low padding");
    }

    // 4) Fail if the linalg_ext::PadOps padding values do not match.
    Attribute innerAttr, outerAttr;
    Value innerValue = padOp.getConstantPaddingValue();
    Value outerValue = outerPadOp.getConstantPaddingValue();
    if (!innerValue || !outerValue ||
        !matchPattern(innerValue, m_Constant(&innerAttr)) ||
        !matchPattern(outerValue, m_Constant(&outerAttr)) ||
        innerAttr != outerAttr) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold PadOps with different padding values");
    }

    // 5) Fail if a dimension is padded by both linalg_ext::PadOps.
    llvm::SmallBitVector innerDims = padOp.getPaddedDims();
    llvm::SmallBitVector outerDims = outerPadOp.getPaddedDims();
    if (innerDims.anyCommon(outerDims)) {
      return rewriter.notifyMatchFailure(
          padOp, "cannot fold PadOps with common padding dimensions");
    }

    // 6) Combine the offsets of the two tensor::ExtractSliceOps. Find the
    // zero-offset and zero-padding tensor::ExtractSliceOp, linalg_ext::PadOp
    // pair for every dimension, and use the offset the other pair. Fail if no
    // zero-offset and zero-padding tensor::ExtractSliceOp, linalg_ext::PadOp
    // pair exists.
    SmallVector<OpFoldResult> newOffsets(rank, rewriter.getIndexAttr(0));
    for (const auto &en : enumerate(newOffsets)) {
      OpFoldResult innerOffset = innerSliceOp.getMixedOffsets()[en.index()];
      OpFoldResult outerOffset = outerSliceOp.getMixedOffsets()[en.index()];
      if (!innerDims.test(en.index()) &&
          (getConstantIntValue(innerOffset) == static_cast<int64_t>(0))) {
        en.value() = outerOffset;
        continue;
      }
      if (!outerDims.test(en.index()) &&
          (getConstantIntValue(outerOffset) == static_cast<int64_t>(0))) {
        en.value() = innerOffset;
        continue;
      }
      return rewriter.notifyMatchFailure(
          padOp, "cannot find zero-offset and zero-padding pair");
    }

    // 7) Combine the sizes of the two tensor::ExtractSliceOps. Take the size of
    // the outer tensor::ExtractSliceOp for the dimensions padded by the outer
    // linalg_ext::PadOp and fail if the size of the inner
    // tensor::ExtractSliceOp does not match the size of the padded dimension.
    // Otherwise, take the size of the inner tensor::ExtractSliceOp.
    SmallVector<OpFoldResult> newSizes = innerSliceOp.getMixedSizes();
    for (const auto &en : enumerate(newSizes)) {
      if (!outerDims.test(en.index()))
        continue;
      OpFoldResult sliceSize = innerSliceOp.getMixedSizes()[en.index()];
      int64_t sourceSize = innerSliceOp.getSourceType().getShape()[en.index()];
      assert(!ShapedType::isDynamic(sourceSize) &&
             "expected padded dimension to have a static size");
      if (getConstantIntValue(sliceSize) != sourceSize) {
        return rewriter.notifyMatchFailure(
            padOp, "cannot fold since the inner ExtractSliceOp size does not "
                   "match the size of the outer padding");
      }
      en.value() = outerSliceOp.getMixedSizes()[en.index()];
    }

    // Combine the high paddings of the two linalg_ext::PadOps.
    SmallVector<OpFoldResult> newHighPad(rank, rewriter.getIndexAttr(0));
    for (const auto &en : enumerate(newHighPad)) {
      if (innerDims.test(en.index()))
        newHighPad[en.index()] = padOp.getMixedHighPad()[en.index()];
      if (outerDims.test(en.index()))
        newHighPad[en.index()] = outerPadOp.getMixedHighPad()[en.index()];
    }

    auto loc = padOp.getLoc();
    // Create a new tensor::ExtractSliceOp, linalg_ext::PadOp pair that performs
    // the two paddings in one step.
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, outerSliceOp.getSource(), newOffsets, newSizes,
        innerSliceOp.getMixedStrides());
    auto resultTy = padOp->getResultTypes().front().cast<ShapedType>();
    llvm::ArrayRef<int64_t> staticShapes = resultTy.getShape();
    SmallVector<Value> dynamicShapes;
    SmallVector<Value, 8> dynShape;
    auto low = padOp.getMixedLowPad();
    for (int i = 0; i < rank; i++) {
      if (ShapedType::isDynamic(staticShapes[i])) {
        auto cstIndex = rewriter.create<arith::ConstantIndexOp>(loc, i);
        auto srcDim =
            rewriter.create<tensor::DimOp>(loc, padOp.input(), cstIndex);
        auto lowV = low[i];
        Value lowValue;
        if (auto attr = lowV.dyn_cast<Attribute>()) {
          if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>()) {
            lowValue = rewriter.create<arith::ConstantIndexOp>(
                loc, intAttr.getValue().getSExtValue());
          }
        } else {
          lowValue = lowV.get<Value>();
        }
        auto highV = newHighPad[i];
        Value highValue;
        if (auto attr = highV.dyn_cast<Attribute>()) {
          if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>()) {
            highValue = rewriter.create<arith::ConstantIndexOp>(
                loc, intAttr.getValue().getSExtValue());
          }
        } else {
          highValue = highV.get<Value>();
        }
        Value dstDim = rewriter.create<arith::AddIOp>(loc, srcDim, lowValue);
        dstDim = rewriter.create<arith::AddIOp>(loc, dstDim, highValue);
        dynamicShapes.push_back(dstDim);
      }
    }
    auto initTensor = rewriter.create<tensor::EmptyOp>(
        loc, staticShapes, resultTy.getElementType(), dynamicShapes);
    auto newPadOp =
        rewriter.create<PadOp>(padOp.getLoc(), newSliceOp.getResult(),
                               initTensor, padOp.getPvalue(), low, newHighPad);
    rewriter.inlineRegionBefore(padOp.getRegion(), newPadOp.getRegion(),
                                newPadOp.getRegion().begin());
    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

// Fold linalg_ext.pad when padding is static zeros.
struct FoldStaticZeroPadding : public OpRewritePattern<linalg_ext::PadOp> {
  using OpRewritePattern<linalg_ext::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg_ext::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (!op.hasZeroLowPad() || !op.hasZeroHighPad())
      return failure();
    rewriter.replaceOp(op, op.input());
    return success();
  }
};

/// Fold static padding size.
struct FoldStaticPadding final : public OpRewritePattern<linalg_ext::PadOp> {
  using OpRewritePattern<linalg_ext::PadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg_ext::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto dynamicLow = op.getLow();
    auto dynamicHigh = op.getHigh();
    auto staticLow = op.getStaticLow();
    auto staticHigh = op.getStaticHigh();

    // Fold constant padding size.
    if (llvm::all_of(dynamicLow,
                     [&](Value v) {
                       return isa<BlockArgument>(v) ||
                              !isa<arith::ConstantOp>(v.getDefiningOp());
                     }) &&
        llvm::all_of(dynamicHigh, [&](Value v) {
          return isa<BlockArgument>(v) ||
                 !isa<arith::ConstantOp>(v.getDefiningOp());
        }))
      return failure();

    int rank = staticLow.size();
    SmallVector<OpFoldResult> newLows, newHighs;
    newLows.reserve(rank);
    newHighs.reserve(rank);

    auto canonicalizePadSize = [&](llvm::ArrayRef<int64_t> staticPart,
                                   ValueRange dynamicPart,
                                   SmallVector<OpFoldResult> &result) {
      for (int i = 0, j = 0; i < rank; i++) {
        if (staticPart[i] == ShapedType::kDynamic) {
          result.push_back(getAsOpFoldResult(dynamicPart[j]));
          j++;
        } else {
          result.push_back(rewriter.getIndexAttr(staticPart[i]));
        }
      }
    };

    canonicalizePadSize(staticLow, dynamicLow, newLows);
    canonicalizePadSize(staticHigh, dynamicHigh, newHighs);

    auto newPad = rewriter.create<linalg_ext::PadOp>(
        op->getLoc(), op.input(), op.getInit(), op.getPvalue(), newLows,
        newHighs);
    rewriter.replaceOp(op, newPad->getResults());
    return success();
  }
};

} // namespace
void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldStaticZeroPadding, FoldStaticPadding, FoldOrthogonalPaddings>(
      context);
}

Value PadOp::getConstantPaddingValue() {
  Value padValue = getPvalue();
  // Check if pad value is a constant.
  if (matchPattern(padValue, m_Constant()))
    return padValue;
  // Check if pad value is defined inside the PadOp block.
  if (padValue.getParentBlock() == &getRegion().front())
    return {};
  // Else: pad value defined outside of the PadOp block.
  return padValue;
}

llvm::SmallBitVector PadOp::getPaddedDims() {
  llvm::SmallBitVector paddedDims(getInputType().getRank());
  auto extractPaddedDims = [&](ArrayRef<OpFoldResult> paddingWidths) {
    for (const auto &en : enumerate(paddingWidths))
      if (getConstantIntValue(en.value()) != static_cast<int64_t>(0))
        paddedDims.set(en.index());
  };
  extractPaddedDims(getMixedLowPad());
  extractPaddedDims(getMixedHighPad());
  return paddedDims;
}

RankedTensorType PadOp::inferResultType(RankedTensorType inputType,
                                        ArrayRef<int64_t> staticLow,
                                        ArrayRef<int64_t> staticHigh,
                                        ArrayRef<int64_t> resultShape) {
  unsigned rank = inputType.getRank();
  assert(staticLow.size() == rank && "unexpected staticLow size mismatch");
  assert(staticHigh.size() == rank && "unexpected staticHigh size mismatch");
  assert((resultShape.empty() || resultShape.size() == rank) &&
         "unexpected resultShape size mismatch");
  SmallVector<int64_t, 4> inferredShape;
  for (auto i : llvm::seq<unsigned>(0, rank)) {
    if (inputType.isDynamicDim(i) || staticLow[i] == ShapedType::kDynamic ||
        staticHigh[i] == ShapedType::kDynamic) {
      inferredShape.push_back(resultShape.empty() ? ShapedType::kDynamic
                                                  : resultShape[i]);
    } else {
      int64_t size = inputType.getDimSize(i) + staticLow[i] + staticHigh[i];
      assert((resultShape.empty() || size == resultShape[i] ||
              resultShape[i] == ShapedType::kDynamic) &&
             "mismatch between inferred shape and result shape");
      inferredShape.push_back(size);
    }
  }
  return RankedTensorType::get(inferredShape, inputType.getElementType());
}
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Dialect/Tensor/IR/TensorOps.cpp
//===----------------------------------------------------------------------===//
void PadOp::build(OpBuilder &builder, OperationState &result, Value input,
                  Value init, Value pvalue, ArrayRef<OpFoldResult> lows,
                  ArrayRef<OpFoldResult> highs,
                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticLows, staticHighs;
  SmallVector<Value> dynamicLows, dynamicHighs;
  dispatchIndexOpFoldResults(lows, dynamicLows, staticLows);
  dispatchIndexOpFoldResults(highs, dynamicHighs, staticHighs);
  auto resultType = init.getType();
  assert(resultType.isa<ShapedType>());
  result.addOperands(input);
  result.addOperands(init);
  result.addOperands(pvalue);
  result.addOperands(dynamicLows);
  result.addOperands(dynamicHighs);
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(
                          {1, 1, 1, static_cast<int32_t>(dynamicLows.size()),
                           static_cast<int32_t>(dynamicHighs.size())}));
  result.addAttribute(getStaticLowAttrName(result.name),
                      builder.getDenseI64ArrayAttr(staticLows));
  result.addAttribute(getStaticHighAttrName(result.name),
                      builder.getDenseI64ArrayAttr(staticHighs));
  (void)result.addRegion();
  if (llvm::isa<RankedTensorType>(resultType))
    result.addTypes(resultType);
  result.addAttributes(attrs);
  // Default region.
  buildIdentityRegion<linalg_ext::ExtYieldOp>(
      builder, result.location, *result.regions.front(), {}, {init});
}

void PadOp::build(OpBuilder &builder, OperationState &result, Value input,
                  Value init, Value pvalue, ValueRange lows, ValueRange highs,
                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> lowValues = llvm::to_vector<4>(
      llvm::map_range(lows, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> highValues = llvm::to_vector<4>(
      llvm::map_range(highs, [](Value v) -> OpFoldResult { return v; }));
  build(builder, result, input, init, pvalue, lowValues, highValues, attrs);
}

void PadOp::build(OpBuilder &builder, OperationState &result, Value input,
                  Value init, Value pvalue, ArrayRef<int64_t> lows,
                  ArrayRef<int64_t> highs, ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> lowValues =
      llvm::to_vector<4>(llvm::map_range(lows, [&](int64_t v) -> OpFoldResult {
        return builder.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> highValues =
      llvm::to_vector<4>(llvm::map_range(highs, [&](int64_t v) -> OpFoldResult {
        return builder.getI64IntegerAttr(v);
      }));
  build(builder, result, input, init, pvalue, lowValues, highValues, attrs);
}

LogicalResult PadOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

void PadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(),
                        getDpsInits());
}

LogicalResult PadOp::verify() {
  Operation *op = getOperation();
  auto inputType = getInputType();
  auto inputRank = inputType.getRank();
  auto initType = getInitType();
  auto initRank = initType.getRank();
  if (inputRank != initRank) {
    return op->emitOpError("expected same rank of input and init");
  }
  if (getStaticLow().size() != inputRank) {
    return op->emitOpError("expected same size of static_low and input");
  }
  if (getStaticHigh().size() != inputRank) {
    return op->emitOpError("expected same size of static_high and input");
  }
  if (inputType.getElementType() != initType.getElementType()) {
    return op->emitOpError("expected same element type of input and init");
  }
  if (inputType.getElementType() != getPaddingValueType()) {
    return op->emitOpError(
        "expected same type of padding value and input elements");
  }
  RankedTensorType sourceType, resultType;
  if (inputType.isa<RankedTensorType>() && initType.isa<RankedTensorType>()) {
    sourceType = inputType.cast<RankedTensorType>();
    resultType = initType.cast<RankedTensorType>();
  } else if (inputType.isa<MemRefType>() && initType.isa<MemRefType>()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    sourceType = RankedTensorType::get(inputShape, inputType.getElementType());
    ArrayRef<int64_t> initShape = initType.getShape();
    resultType = RankedTensorType::get(initShape, initType.getElementType());
  }
  auto expectedType =
      PadOp::inferResultType(sourceType, getStaticLow(), getStaticHigh());
  for (int i = 0, e = sourceType.getRank(); i < e; ++i) {
    if (resultType.getDimSize(i) == expectedType.getDimSize(i))
      continue;
    if (expectedType.isDynamicDim(i))
      continue;
    if (resultType.isDynamicDim(i))
      continue;
    return emitError("specified type ")
           << resultType << " does not match the inferred type " << expectedType
           << " on dimension " << i;
  }
  return success();
}

/////// Operations corresponding to library calls defined with Tablegen ////////
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"

#define GET_OP_CLASSES
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtStructedOps.cpp.inc"

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

Operation *LinalgExtDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
