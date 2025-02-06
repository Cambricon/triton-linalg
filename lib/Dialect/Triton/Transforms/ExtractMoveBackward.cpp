//===- ExtractMoveBackward.cpp - Move backward extract op -------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <stddef.h>
#include <stdint.h>
#include <utility>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

// IWYU pragma: begin_keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "triton-linalg/Dialect/Triton/Transforms/PassDetail.h"
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::triton;

/// Check whether two arrays are deterministic equal.
static bool isEqualConstantIntOrValue(ArrayRef<OpFoldResult> lhs,
                                      ArrayRef<OpFoldResult> rhs) {
  if (lhs.size() != rhs.size())
    return false;

  for (size_t i = 0, e = lhs.size(); i < e; ++i) {
    if (isEqualConstantIntOrValue(lhs[i], rhs[i]))
      continue;

    return false;
  }

  return true;
}

/// Check whether the result rank is less than input.
static inline bool isOutputRankReduced(tensor::ExtractSliceOp op) {
  int64_t srcRank = op.getSourceType().getRank();
  int64_t dstRank = op.getResultType().getRank();
  return srcRank > dstRank;
}

/// Check whether ofr has same size with the `dim` of `value`. Current
/// pessimistic implementation supports:
/// 1. ofr and the dim of value are static and same.
/// 2. ofr is a value and defined by tensor.dim, with the index `dim` and
/// source `value`.
static bool hasSameSizeWithDim(OpFoldResult ofr, Value value, int64_t dim) {
  auto type = mlir::dyn_cast<ShapedType>(value.getType());
  assert(type && dim >= 0 && dim < type.getRank() &&
         "Expected value with ShapedType and dim is a valid axis index.");

  // Check whether the dim and ofr are static and same.
  auto intOFR = getConstantIntValue(ofr);
  if (!type.isDynamicDim(dim) && intOFR && *intOFR == type.getShape()[dim])
    return true;

  // Check whether ofr is defined by an tensor.dim, with the index `dim` and
  // source `value`.
  auto ofrValue = mlir::dyn_cast<Value>(ofr);
  auto dimOp = ofrValue ? ofrValue.getDefiningOp<tensor::DimOp>() : nullptr;
  if (!dimOp || dimOp.getSource() != value)
    return false;

  Value index = dimOp.getIndex();
  auto constantOp = index.getDefiningOp<arith::ConstantOp>();
  return constantOp &&
         mlir::cast<mlir::IntegerAttr>(constantOp.getValue()).getInt() == dim;
}

/// Check at most only one of the dims of src reassociation postions is not one.
static bool hasAtMostOneDimNonTrivial(
    SmallVector<ReassociationIndices, 4> &reassociationIndices,
    int64_t dstDimIdx, const Value &collapseSrc, PatternRewriter &rewriter,
    const Location &loc, int64_t srcDimCntOffset) {
  int64_t cntDimOne = 0;
  for (size_t i = 0; i < reassociationIndices[dstDimIdx].size(); ++i) {
    auto constDim = getConstantIntValue(
        getDim(rewriter, loc, collapseSrc, srcDimCntOffset + i));
    if (constDim && *constDim == 1) {
      ++cntDimOne;
    }
  }
  return (cntDimOne >= reassociationIndices[dstDimIdx].size() - 1);
}

/// Reshape input to resultType by adding unit dims.
///
/// Example: Support we want to reshape %0 with type tensor<1x16x1x8xf32>
/// to tensor<1x16x1x1x8x1xf32>:
///
/// ```mlir
///   %res = tensor.expand_shape %0 [[0], [1], [2, 3], [4, 5]]
///        : tensor<1x16x1x8xf32> into tensor<1x16x1x1x8x1xf32>
/// ```
static Value expandShapeToResultTypeByAddUnitDims(OpBuilder &b, Location loc,
                                                  ShapedType resultType,
                                                  Value value) {
  auto sourceType = mlir::cast<ShapedType>(value.getType());
  int64_t dstRank = resultType.getRank();
  int64_t srcRank = sourceType.getRank();
  int64_t rankDiff = dstRank - srcRank;
  assert(rankDiff >= 0 && "unsupport reshape to result type with smaller rank");
  if (rankDiff == 0)
    return value;

  auto srcShape = sourceType.getShape();
  auto dstShape = resultType.getShape();
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<int64_t> leadingInconsistentDims;
  for (unsigned srcPos = 0, dstPos = 0; dstPos < dstRank; ++dstPos) {
    // Handle unchanged srcPos.
    if (srcPos < srcRank && srcShape[srcPos] == dstShape[dstPos]) {
      reassociation.push_back({dstPos});
      ++srcPos;
      continue;
    }

    // Set reassociation to empty if srcRank is 0.
    if (srcRank == 0)
      break;

    // When reassociation is empty, the leading dims need to be assigned to
    // latter groups.
    if (reassociation.empty()) {
      leadingInconsistentDims.push_back(dstPos);
      continue;
    }

    // Assign dims to the former groups.
    reassociation.back().push_back(dstPos);
  }

  if (!leadingInconsistentDims.empty() && !reassociation.empty())
    reassociation.front().insert(reassociation.front().begin(),
                                 leadingInconsistentDims.begin(),
                                 leadingInconsistentDims.end());

  return b.create<tensor::ExpandShapeOp>(loc, resultType, value, reassociation);
}

/// Reshape input to resultType by dropping unit dims.
///
/// Example: Support we want to reshape %0 with type tensor<1x16x1x1x8x1xf32>
/// to tensor<1x16x1x8xf32>:
///
/// ```mlir
///   %res = tensor.collapse_shape %0 [[0], [1], [2, 3], [4, 5]]
///        : tensor<1x16x1x1x8x1xf32> into tensor<1x16x1x8xf32>
/// ```
static Value reshapeToResultTypeByDropUnitDims(OpBuilder &b, Location loc,
                                               ShapedType resultType,
                                               Value value) {
  auto sourceType = mlir::cast<ShapedType>(value.getType());
  int64_t dstRank = resultType.getRank();
  int64_t srcRank = sourceType.getRank();
  int64_t rankDiff = srcRank - dstRank;
  assert(rankDiff >= 0 && "unsupport reshape to result type with larger rank");
  if (rankDiff == 0)
    return value;

  auto srcShape = sourceType.getShape();
  auto dstShape = resultType.getShape();
  SmallVector<int64_t> leadingInconsistentDims;
  SmallVector<ReassociationIndices> reassociation;
  for (unsigned srcPos = 0, dstPos = 0; srcPos < srcRank; ++srcPos) {
    // Handle unchanged srcPos.
    if (dstPos < dstRank && srcShape[srcPos] == dstShape[dstPos]) {
      reassociation.push_back({srcPos});
      ++dstPos;
      continue;
    }

    // Set reassociation to empty if dstRank is 0.
    if (dstRank == 0)
      break;

    // When reassociation is empty, the leading dims need to be assigned to
    // latter groups.
    if (reassociation.empty()) {
      leadingInconsistentDims.push_back(srcPos);
      continue;
    }

    // Assign dims to the former groups.
    reassociation.back().push_back(srcPos);
  }

  if (!leadingInconsistentDims.empty() && !reassociation.empty())
    reassociation.front().insert(reassociation.front().begin(),
                                 leadingInconsistentDims.begin(),
                                 leadingInconsistentDims.end());
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(value);
  return b.create<tensor::CollapseShapeOp>(loc, value, reassociation);
}

namespace {
enum class ExtractType {
  EXTRACT,
  EXTRACTSLICE,
};

/// Data structures are utilized to facilitate the extraction of information
/// during extraction operations. The meaning of each field is outlined
/// as follows:
/// - `offsets` represents the offsets or indices to extract, used for
/// tensor.extract and tensor.extract_slice.
/// - `sizes` represents sizes for each dimension, used for tensor.extract_slice
/// only.
/// - `strides` represents strides for each dimension, used for
/// tensor.extract_slice only.
/// - `extractedVal` represents the extracted result.
struct ExtractState {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  ExtractType type;
  /// In the beginning of moving extract-like operations backward, the
  /// extractedVal field is initialized to the result of original
  /// extract-like operations. Then, in the process of moving extract-slice
  /// backward, extraction state fields are updated when visiting operations,
  /// and the extractedVal field are updated in reverse order.
  /// Take the following case as an example:
  /// ```
  ///      Value
  ///        |
  ///       op-n
  ///        |
  ///       ...
  ///        |
  ///       op-1
  ///        |
  /// extract-like op
  /// ```
  /// The extraction updating process is:
  /// ```
  ///      Value               Value                      Value
  ///        |                   |               (on+1, sn+1, s'n+1, null)
  ///        |                   |                          |
  ///       op-n                op-n                       op-n
  ///        |        ->         |    -> ... ->      (on, sn, s'n, null)
  ///        |                   |                          |
  ///       ...                 ...                        ...
  ///        |                   |                          |
  ///       op-1                op-1                       op-1
  ///        |            (o1, s1, s'1, null)        (o1, s1, s'1, null)
  ///        |                   |                          |
  /// extract-like op     extract-like op            extract-like op
  /// (o0, s0, s'0, ev0)  (o0, s0, s'0, ev0)         (o0, s0, s'0, ev0)
  ///
  ///      Value                               Value
  ///  extract-like op'                    extract-like op'
  /// (on+1, sn+1, s'n+1, null)  ->   (on+1, sn+1, s'n+1, evn+1')
  ///                                           |
  ///                                         op-n'
  ///                                      (on, sn, s'n, evn')
  //
  ///             Value
  ///         extract-like op'
  ///     (on+1, sn+1, s'n+1, evn+1')
  ///              |
  ///            op-n'
  ///   ->   (on, sn, s'n, evn')
  ///              |
  ///             ...
  ///              |
  ///             op-1'
  ///        (o1, s1, s'1, ev1')
  /// ```
  Value extractedVal;

  ExtractState() = default;
  ExtractState(tensor::ExtractOp op);
  ExtractState(tensor::ExtractSliceOp op);
  ExtractState copyWithoutValue();

  /// Check whether `state` is same as this.
  bool isSameExceptVal(const ExtractState &state);
  /// Specialization version for operations.
  bool isSameExceptVal(tensor::ExtractOp op);
  bool isSameExceptVal(tensor::ExtractSliceOp op);
};

bool ExtractState::isSameExceptVal(const ExtractState &state) {
  return isEqualConstantIntOrValue(offsets, state.offsets) &&
         isEqualConstantIntOrValue(sizes, state.sizes) &&
         isEqualConstantIntOrValue(strides, state.strides) &&
         type == state.type;
}

bool ExtractState::isSameExceptVal(tensor::ExtractOp op) {
  return isSameExceptVal(ExtractState{op});
}

bool ExtractState::isSameExceptVal(tensor::ExtractSliceOp op) {
  return isSameExceptVal(ExtractState{op});
}

ExtractState ExtractState::copyWithoutValue() {
  ExtractState state;
  state = *this;
  state.extractedVal = Value();
  return state;
}

ExtractState::ExtractState(tensor::ExtractOp op)
    : offsets(getAsOpFoldResult(op.getIndices())), type(ExtractType::EXTRACT),
      extractedVal(op.getResult()) {}

ExtractState::ExtractState(tensor::ExtractSliceOp op)
    : offsets(op.getMixedOffsets()), sizes(op.getMixedSizes()),
      strides(op.getMixedStrides()), type(ExtractType::EXTRACTSLICE),
      extractedVal(op.getResult()) {}

} // anonymous namespace

/// Get the insertion point by analysing values that will be used.
static OpBuilder::InsertPoint getInsertionPoint(ArrayRef<Value> values,
                                                ExtractState &state,
                                                PatternRewriter &rewriter) {
  // Collect all related values.
  SmallVector<OpFoldResult> opFoldResults;
  opFoldResults.append(state.offsets);
  opFoldResults.append(state.sizes);
  opFoldResults.append(state.strides);
  std::pair<SmallVector<int64_t>, SmallVector<Value>> attrOrVals =
      decomposeMixedValues(opFoldResults);
  SmallVector<Value> dependentVals = attrOrVals.second;
  dependentVals.append(SmallVector<Value>(values));

  // Create current func op dominance info.
  auto funcOp = dependentVals.front()
                    .getParentRegion()
                    ->getParentOfType<FunctionOpInterface>();
  DominanceInfo domInfo(funcOp);
  // Divide values by different blocks.
  DenseMap<Block *, SmallVector<Value>> blockVals;
  for (auto val : dependentVals) {
    Block *block = val.getParentBlock();
    if (blockVals.count(block)) {
      blockVals[block].emplace_back(val);
    } else {
      blockVals.insert({block, SmallVector<Value>{val}});
    }
  }
  // Find the innermost block according to dominance info. Note: here all
  // related values will be used to create operations, so their defining block
  // is locating at the same branch of dominance tree. Global variable is not
  // considered.
  Block *innerBlock = blockVals.begin()->first;
  for (auto &[key, value] : blockVals) {
    if (domInfo.dominates(innerBlock, key)) {
      innerBlock = key;
    }
  }
  // Find the last value in the innermost block.
  SmallVector<Value> innerVals = blockVals[innerBlock];
  SmallVector<Operation *> ops;
  SmallVector<Value> args;
  for (auto val : innerVals) {
    auto op = val.getDefiningOp();
    if (op) {
      ops.emplace_back(op);
    } else {
      args.emplace_back(val);
    }
  }
  // If op exists, find the last one as the insertion point.
  if (!ops.empty()) {
    llvm::sort(
        ops, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    return OpBuilder::InsertPoint(innerBlock, ++Block::iterator(ops.back()));
  }
  // Otherwise, return one argument as the insertion point.
  return OpBuilder::InsertPoint(
      innerBlock, mlir::cast<BlockArgument>(args.front()).getOwner()->begin());
}

static void getExtractedValueFrom(Value value, ExtractState &state,
                                  Location loc, PatternRewriter &rewriter) {
  if (state.extractedVal)
    return;
  rewriter.restoreInsertionPoint(getInsertionPoint({value}, state, rewriter));
  // Get value by tensor.extract.
  if (state.type == ExtractType::EXTRACT) {
    state.extractedVal = rewriter.create<tensor::ExtractOp>(
        loc, value, vector::getAsValues(rewriter, loc, state.offsets));
    return;
  }

  // Get value by tensor.extract_slice.
  state.extractedVal = rewriter.create<tensor::ExtractSliceOp>(
      loc, value, state.offsets, state.sizes, state.strides);
}

namespace {

//===----------------------------------------------------------------------===//
// Extract Analysis
//===----------------------------------------------------------------------===//
/// An utility class used to analyze the generation path of an extracted value,
/// and rewrite by operations with less computational effort. The general
/// process is roughly:
/// 1. Initialize the input operand of tensor.extract with offsets.
/// 2. Backtrace ancestor operations and pass extract state information to these
/// operations' inputs until operands defined by unsupported operations or as
/// block arguments, namely start operands.
/// 3. Inserting tensor.extract or tensor.extract_slice to get extracted value
/// of start operands.
/// 4. Inserting identity semantic operations to calculate original extract
/// value.
///
/// For example, consider the following examples:
///
/// Case 1.
/// ``` mlir
/// func.func @extract(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
///   %c0 = arith.constant 0 : index
///   %0 = arith.addi %arg0, %arg1 : tensor<128xi32>
///   %1 = tensor.extract %0[%c0] : tensor<128xi32>
///   return %1 : i32
/// }
/// ```
///
/// After running, we get the expected:
///
/// ``` mlir
/// func.func @extract(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>) -> i32 {
///   %c0 = arith.constant 0 : index
///   %0 = tensor.extract %arg0[%c0] : tensor<128xi32>
///   %1 = tensor.extract %arg1[%c0] : tensor<128xi32>
///   %2 = arith.addi %0, %1 : i32
///   return %2 : i32
/// }
/// ```
///
/// Case 2.
/// ``` mlir
/// func.func @extract_slice(%arg0: tensor<128x16xi32>,
///                          %arg1: tensor<128x16xi32>) -> tensor<16x4xi32> {
///   %0 = tensor.empty() : tensor<128x16xi32>
///   %1 = linalg.map { arith.addi }
///        ins(%arg0, %arg1 : tensor<128x16xi32>, tensor<128x16xi32>)
///        outs(%0 : tensor<128x16xi32>)
///   %2 = tensor.extract_slice %1[0, 0] [16, 4] [2, 2]
///        : tensor<128x16xi32> to tensor<16x4xi32>
///   return %2 : tensor<16x4xi32>
/// }
/// ```
///
/// After running, we get the expected:
///
/// ``` mlir
/// func.func @extract_slice(%arg0: tensor<128x16xi32>,
///                          %arg1: tensor<128x16xi32>) -> tensor<16x4xi32> {
///   %extracted_slice = tensor.extract_slice %arg0[0, 0] [16, 4] [2, 2]
///                      : tensor<128x16xi32> to tensor<16x4xi32>
///   %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [16, 4] [2, 2]
///                        : tensor<128x16xi32> to tensor<16x4xi32>
///   %0 = tensor.empty() : tensor<16x4xi32>
///   %mapped = linalg.map { arith.addi }
///             ins(%extracted_slice, %extracted_slice_0
///                 : tensor<16x4xi32>, tensor<16x4xi32>)
///             outs(%0 : tensor<16x4xi32>)
///   return %mapped : tensor<16x4xi32>
/// }
/// ```
class ExtractAnalysis {
public:
  /// Greedily moving extract-like operations ahead and rewrite
  /// the computation procedure to scalar operations.
  template <typename OpTy>
  static LogicalResult rewriteExtractLikeOp(OpTy op, PatternRewriter &rewriter);

  /// Call the corresponding function based on the defining operation
  /// to generate an equivalent semantic computation path.
  static void visitOperand(Value operand, ExtractState &state,
                           Operation *startOp, Location loc,
                           PatternRewriter &rewriter);

private:
  //===------------------------------------------------------------------===//
  // Helper functions to passthrough supported operations and
  // set extract states.
  //===------------------------------------------------------------------===//
  template <class OpTy>
  static void visitOperandFromOp(OpTy op, ExtractState &state, Location loc,
                                 PatternRewriter &rewriter);
};

template <>
void ExtractAnalysis::visitOperandFromOp(linalg::MapOp op, ExtractState &state,
                                         Location loc,
                                         PatternRewriter &rewriter) {
  SmallVector<ExtractState> operandStates;
  operandStates.reserve(op.getNumDpsInputs());
  for (Value v : op.getInputs()) {
    auto operandState = state.copyWithoutValue();
    visitOperand(v, operandState, op, loc, rewriter);
    operandStates.push_back(operandState);
  }

  // Retrieve extracted operands.
  SmallVector<Value> foldOperands(llvm::map_range(
      operandStates, [](const ExtractState &s) { return s.extractedVal; }));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint(foldOperands, state, rewriter));
  if (state.type == ExtractType::EXTRACT) {
    // Map operands to extracted operands.
    IRMapping bvm;
    Block *body = op.getBody();
    for (const auto &en : llvm::enumerate(op.getOpOperandsMatchingBBargs())) {
      OpOperand *opOperand = en.value();
      BlockArgument bbarg = body->getArgument(opOperand->getOperandNumber());
      bvm.map(bbarg, foldOperands[en.index()]);
    }

    // Clone map body to generate extracted map result.
    for (auto &payload : body->getOperations()) {
      if (!mlir::isa<linalg::YieldOp>(payload))
        rewriter.clone(payload, bvm);
    }
    auto &yieldOp = body->getOperations().back();
    state.extractedVal = bvm.lookupOrDefault(yieldOp.getOperand(0));
    return;
  }

  // Extract slice input operands of original map op and clone the map op with
  // updated operands.
  auto newInit = rewriter.create<tensor::EmptyOp>(
      loc, state.sizes, getElementTypeOrSelf(op.getInit().getType()));
  auto newMapOp =
      rewriter.create<linalg::MapOp>(loc, foldOperands, newInit, nullptr);
  rewriter.cloneRegionBefore(op.getMapper(), newMapOp.getMapper(),
                             newMapOp.getMapper().begin());
  state.extractedVal = newMapOp->getResult(0);
}

template <>
void ExtractAnalysis::visitOperandFromOp(linalg::FillOp op, ExtractState &state,
                                         Location loc,
                                         PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op.getResult(0)}, state, rewriter));
  if (state.type == ExtractType::EXTRACT) {
    state.extractedVal = op.getOperand(0);
    return;
  }
  Value output = rewriter.create<tensor::EmptyOp>(
      loc, state.sizes,
      mlir::cast<ShapedType>(op.getResult(0).getType()).getElementType());
  state.extractedVal =
      rewriter.create<linalg::FillOp>(loc, op.getOperand(0), output)
          .getResult(0);
}

template <>
void ExtractAnalysis::visitOperandFromOp(linalg_ext::MakeRangeOp op,
                                         ExtractState &state, Location loc,
                                         PatternRewriter &rewriter) {
  assert(state.offsets.size() == 1 &&
         "Offset size must be 1 for make_range op.");
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op.getResult(0)}, state, rewriter));
  if (state.type == ExtractType::EXTRACT) {
    Value offset = rewriter.create<arith::IndexCastOp>(
        loc, op.getStart().getType(),
        getValueOrCreateConstantIndexOp(rewriter, loc, state.offsets[0]));
    state.extractedVal =
        rewriter.create<arith::AddIOp>(loc, offset, op.getStart());
    return;
  }

  // Since linalg_ext.make_range is the end point of value to extract,
  // There are several advantages to keep make_range unchanged.
  // 1. The size of make_range is the power of 2, and the hardware
  // implementation can be vectorized complemently.
  // 2. Without the constraint of sliced size, there are more opportunities
  // to reuse the result of make_range.
  // 3. If the slice size is dynamic, it would allocate a memory space with
  // same size as original make_range. No memory usage would be reduced.
  // For performance consideration, skip linalg_ext.make_range.
  return getExtractedValueFrom(op.getResult(0), state, loc, rewriter);
}

template <>
void ExtractAnalysis::visitOperandFromOp(linalg::BroadcastOp op,
                                         ExtractState &state, Location loc,
                                         PatternRewriter &rewriter) {
  ArrayRef<int64_t> dimensions = op.getDimensions();
  ExtractState operandState;
  operandState.type = state.type;
  // Derive the extract state of linalg_ext.broadcast input operand.
  for (const auto &it : llvm::enumerate(state.offsets)) {
    if (llvm::find(dimensions, it.index()) == dimensions.end()) {
      operandState.offsets.push_back(it.value());
      if (state.type == ExtractType::EXTRACTSLICE) {
        operandState.sizes.push_back(state.sizes[it.index()]);
        operandState.strides.push_back(state.strides[it.index()]);
      }
    }
  }

  visitOperand(op.getInput(), operandState, op, loc, rewriter);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({operandState.extractedVal}, state, rewriter));
  if (state.type == ExtractType::EXTRACT)
    state.extractedVal = operandState.extractedVal;
  else {
    auto resElemTy = getElementTypeOrSelf(op->getResult(0).getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, state.sizes, resElemTy);
    state.extractedVal =
        rewriter
            .create<linalg::BroadcastOp>(loc, operandState.extractedVal, init,
                                         op.getDimensions())
            ->getResult(0);
  }
}

static void extractFromCollapseShapeOp(tensor::CollapseShapeOp op,
                                       ExtractState &state, Location loc,
                                       PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op->getResult(0)}, state, rewriter));
  Value collapseSrc = op.getSrc();
  auto reassociationIndices = op.getReassociationIndices();
  int64_t resRank = op.getResultType().getRank();
  ExtractState operandState;
  operandState.type = state.type;
  for (int64_t dstDimIdx = 0; dstDimIdx < resRank; ++dstDimIdx) {
    if (reassociationIndices[dstDimIdx].size() == 1) {
      operandState.offsets.push_back(state.offsets[dstDimIdx]);
    } else {
      SmallVector<Value> localIndices;
      Value dstIndices = getValueOrCreateConstantIndexOp(
          rewriter, loc, state.offsets[dstDimIdx]);
      for (auto dim : llvm::reverse(reassociationIndices[dstDimIdx])) {
        // Get source dimension size.
        auto srcSize = getDimValue(rewriter, loc, collapseSrc, dim);
        localIndices.push_back(
            rewriter.createOrFold<arith::RemUIOp>(loc, dstIndices, srcSize));
        dstIndices =
            rewriter.createOrFold<arith::DivUIOp>(loc, dstIndices, srcSize);
      }
      operandState.offsets.append(localIndices.rbegin(), localIndices.rend());
    }
  }

  if (resRank == 0) {
    Value c0 =
        rewriter.createOrFold<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto srcRank = mlir::cast<ShapedType>(collapseSrc.getType()).getRank();
    operandState.offsets.append(srcRank, c0);
  }

  ExtractAnalysis::visitOperand(collapseSrc, operandState, op, loc, rewriter);
  state.extractedVal = operandState.extractedVal;
}

/// Try to bypass tensor.collapse_shape and move tensor.extract_slice op
/// backward. Currently, for each group of collapsed dimensions with association
/// number larger than 1, one of the following conditions must be satisfied:
/// 1. Extracts the entire size of the collapsed dimension.
/// 2. Only extract the first element.
/// 3. At most one of the dims in reassociationIndices is >1.
static void extractSliceFromCollapseShapeOp(tensor::CollapseShapeOp op,
                                            ExtractState &state, Location loc,
                                            PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op->getResult(0)}, state, rewriter));
  Value collapseSrc = op.getSrc();
  Value collapseDst = op.getResult();
  auto reassociationIndices = op.getReassociationIndices();
  int64_t resRank = op.getResultType().getRank();
  ExtractState operandState;
  operandState.type = state.type;

  auto indexAttrZero = rewriter.getIndexAttr(0);
  auto indexAttrOne = rewriter.getIndexAttr(1);
  int64_t srcDimCntOffset = 0;
  for (int64_t dstDimIdx = 0; dstDimIdx < resRank; ++dstDimIdx) {
    if (dstDimIdx >= 1) {
      srcDimCntOffset += reassociationIndices[dstDimIdx - 1].size();
    }
    // Skip dims with association number 1.
    if (reassociationIndices[dstDimIdx].size() == 1) {
      operandState.offsets.push_back(state.offsets[dstDimIdx]);
      operandState.sizes.push_back(state.sizes[dstDimIdx]);
      operandState.strides.push_back(state.strides[dstDimIdx]);
      continue;
    }

    // Meet condition 1.
    if (hasSameSizeWithDim(state.sizes[dstDimIdx], collapseDst, dstDimIdx)) {
      auto dimNum = reassociationIndices[dstDimIdx].size();
      operandState.offsets.append(dimNum, indexAttrZero);
      operandState.strides.append(dimNum, indexAttrOne);
      for (auto dim : reassociationIndices[dstDimIdx])
        operandState.sizes.push_back(getDim(rewriter, loc, collapseSrc, dim));
      continue;
    }

    // Meet condition 2.
    auto intSize = getConstantIntValue(state.sizes[dstDimIdx]);
    auto intOffset = getConstantIntValue(state.offsets[dstDimIdx]);
    if (intOffset && intSize && *intOffset == 0 && *intSize == 1) {
      auto dimNum = reassociationIndices[dstDimIdx].size();
      operandState.offsets.append(dimNum, indexAttrZero);
      operandState.sizes.append(dimNum, indexAttrOne);
      operandState.strides.append(dimNum, indexAttrOne);
      continue;
    }

    // Meet condition 3.
    if (hasAtMostOneDimNonTrivial(reassociationIndices, dstDimIdx, collapseSrc,
                                  rewriter, loc, srcDimCntOffset)) {
      bool statePushed = false;

      for (size_t i = 0; i < reassociationIndices[dstDimIdx].size(); ++i) {
        auto constDim = getConstantIntValue(
            getDim(rewriter, loc, collapseSrc, srcDimCntOffset + i));
        // If all dims at reassociation indices are 1, pass state info to the
        // last dim.
        bool flag =
            (i == reassociationIndices[dstDimIdx].size() - 1 && !statePushed)
                ? false
                : true;
        if (constDim && *constDim == 1 && flag) {
          operandState.offsets.push_back(indexAttrZero);
          operandState.sizes.push_back(indexAttrOne);
          operandState.strides.push_back(indexAttrOne);
        } else {
          operandState.offsets.push_back(state.offsets[dstDimIdx]);
          operandState.sizes.push_back(state.sizes[dstDimIdx]);
          operandState.strides.push_back(state.strides[dstDimIdx]);
          statePushed = true;
        }
      }
      continue;
    }

    // Stop moving backward.
    getExtractedValueFrom(collapseDst, state, loc, rewriter);
    return;
  }

  // If the reassociation is empty, the operand state is constructed based on
  // the rank of source.
  int64_t srcRank = op.getSrcType().getRank();
  if (resRank == 0) {
    operandState.offsets = SmallVector<OpFoldResult>(srcRank, indexAttrZero);
    operandState.sizes = SmallVector<OpFoldResult>(srcRank, indexAttrOne);
    operandState.strides = SmallVector<OpFoldResult>(srcRank, indexAttrOne);
  }

  auto srcTy = mlir::cast<RankedTensorType>(op.getResultType());
  auto resultTy = tensor::ExtractSliceOp::inferResultType(
      srcTy, state.offsets, state.sizes, state.strides);
  ExtractAnalysis::visitOperand(collapseSrc, operandState, op, loc, rewriter);
  rewriter.setInsertionPointAfterValue(operandState.extractedVal);
  state.extractedVal = rewriter.create<tensor::CollapseShapeOp>(
      loc, resultTy, operandState.extractedVal, reassociationIndices);
}

template <>
void ExtractAnalysis::visitOperandFromOp(tensor::CollapseShapeOp op,
                                         ExtractState &state, Location loc,
                                         PatternRewriter &rewriter) {
  if (state.type == ExtractType::EXTRACT)
    extractFromCollapseShapeOp(op, state, loc, rewriter);
  else
    extractSliceFromCollapseShapeOp(op, state, loc, rewriter);
}

static void extractFromExpandShapeOp(tensor::ExpandShapeOp op,
                                     ExtractState &state, Location loc,
                                     PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op->getResult(0)}, state, rewriter));
  Value expandSrc = op.getSrc();
  Value expandDst = op.getResult();
  int64_t srcRank = op.getSrcType().getRank();
  auto reassociationIndices = op.getReassociationIndices();
  int64_t dstDimIdx = 0;
  ExtractState operandState;
  operandState.type = state.type;
  for (auto srcDimIdx : llvm::seq<int64_t>(0, srcRank)) {
    if (reassociationIndices[srcDimIdx].size() == 1) {
      operandState.offsets.push_back(state.offsets[dstDimIdx]);
    } else {
      Value newIndice = getValueOrCreateConstantIndexOp(
          rewriter, loc, state.offsets[dstDimIdx]);
      for (auto dstDim : ArrayRef(reassociationIndices[srcDimIdx].begin(),
                                  reassociationIndices[srcDimIdx].end())
                             .drop_front()) {
        auto dstSize = getDimValue(rewriter, loc, expandDst, dstDim);
        newIndice = rewriter.createOrFold<arith::AddIOp>(
            loc,
            getValueOrCreateConstantIndexOp(rewriter, loc,
                                            state.offsets[dstDim]),
            rewriter.createOrFold<arith::MulIOp>(loc, dstSize, newIndice));
      }
      operandState.offsets.push_back(newIndice);
    }
    dstDimIdx += reassociationIndices[srcDimIdx].size();
  }
  ExtractAnalysis::visitOperand(expandSrc, operandState, op, loc, rewriter);
  state.extractedVal = operandState.extractedVal;
}

/// Try to bypass tensor.expand_shape and move tensor.strided_slice operation
/// backward. Currently, for each group of expanded dimensions with dimnesion
/// number larger than 1, one of the following conditions must be satisfied.
/// 1. Extract all for each dimension.
/// 2. Only extract the first element for each dimension.
static void extractSliceFromExpandShapeOp(tensor::ExpandShapeOp op,
                                          ExtractState &state, Location loc,
                                          PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op->getResult(0)}, state, rewriter));
  Value expandSrc = op.getSrc();
  Value expandDst = op.getResult();
  int64_t srcRank = op.getSrcType().getRank();
  auto reassociationIndices = op.getReassociationIndices();
  ExtractState operandState;
  operandState.type = state.type;
  auto dstType = mlir::cast<ShapedType>(expandDst.getType());
  auto indexAttrZero = rewriter.getIndexAttr(0);
  auto indexAttrOne = rewriter.getIndexAttr(1);
  for (auto srcDimIdx : llvm::seq<int64_t>(0, srcRank)) {
    // Skip dims with association number 1.
    if (reassociationIndices[srcDimIdx].size() == 1) {
      auto dstDimIdx = reassociationIndices[srcDimIdx][0];
      operandState.offsets.push_back(state.offsets[dstDimIdx]);
      operandState.sizes.push_back(state.sizes[dstDimIdx]);
      operandState.strides.push_back(state.strides[dstDimIdx]);
      continue;
    }

    // Meet condition 1.
    bool extractAll = true;
    for (auto dstDim : reassociationIndices[srcDimIdx]) {
      auto intSize = getConstantIntValue(state.sizes[dstDim]);
      if (dstType.isDynamicDim(dstDim) || !intSize ||
          dstType.getShape()[dstDim] != *intSize) {
        extractAll = false;
        break;
      }
    }
    if (extractAll) {
      operandState.offsets.push_back(rewriter.getIndexAttr(0));
      operandState.sizes.push_back(getDim(rewriter, loc, expandSrc, srcDimIdx));
      operandState.strides.push_back(rewriter.getIndexAttr(1));
      continue;
    }

    // Meet condition 2.
    bool extractFirst = true;
    for (auto dstDim : reassociationIndices[srcDimIdx]) {
      auto intOffset = getConstantIntValue(state.offsets[dstDim]);
      auto intSize = getConstantIntValue(state.sizes[dstDim]);
      if (!intOffset || *intOffset != 0 || !intSize || *intSize != 1) {
        extractFirst = false;
        break;
      }
    }
    if (extractFirst) {
      operandState.offsets.push_back(indexAttrZero);
      operandState.sizes.push_back(indexAttrOne);
      operandState.strides.push_back(indexAttrOne);
      continue;
    }

    // Stop moving backward.
    getExtractedValueFrom(expandDst, state, loc, rewriter);
    return;
  }
  auto srcTy = mlir::cast<RankedTensorType>(op.getResultType());
  auto resultTy = tensor::ExtractSliceOp::inferResultType(
      srcTy, state.offsets, state.sizes, state.strides);
  ExtractAnalysis::visitOperand(expandSrc, operandState, op, loc, rewriter);
  rewriter.setInsertionPointAfterValue(operandState.extractedVal);
  state.extractedVal = rewriter.create<tensor::ExpandShapeOp>(
      loc, resultTy, operandState.extractedVal, reassociationIndices);
}

template <>
void ExtractAnalysis::visitOperandFromOp(tensor::ExpandShapeOp op,
                                         ExtractState &state, Location loc,
                                         PatternRewriter &rewriter) {
  if (state.type == ExtractType::EXTRACT)
    extractFromExpandShapeOp(op, state, loc, rewriter);
  else
    extractSliceFromExpandShapeOp(op, state, loc, rewriter);
}

template <>
void ExtractAnalysis::visitOperandFromOp(Operation *op, ExtractState &state,
                                         Location loc,
                                         PatternRewriter &rewriter) {
  auto *dialect = op->getDialect();
  (void)dialect;
  assert((mlir::isa<arith::ArithDialect>(dialect) ||
          mlir::isa<math::MathDialect>(dialect)) &&
         "unregister operations in extact analysis for now.");

  SmallVector<ExtractState> operandStates;
  for (Value v : op->getOperands()) {
    auto operandState = state.copyWithoutValue();
    visitOperand(v, operandState, op, loc, rewriter);
    operandStates.push_back(operandState);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(
      getInsertionPoint({op->getResult(0)}, state, rewriter));

  SmallVector<Value> foldOperands(llvm::map_range(
      operandStates, [](const ExtractState &s) { return s.extractedVal; }));

  // Since `arith.constant` uses attribute to represent value, we can not
  // use operation identifier to update it directly. Here, we utilize the
  // fold methods of extract-like operations to eliminate constant operations.
  if (mlir::isa<arith::ConstantOp>(op))
    return getExtractedValueFrom(op->getResult(0), state, loc, rewriter);

  auto resultTy = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  Type newResTy =
      (state.type == ExtractType::EXTRACT)
          ? resultTy.getElementType()
          : tensor::ExtractSliceOp::inferResultType(resultTy, state.offsets,
                                                    state.sizes, state.strides);

  auto opName = op->getName().getIdentifier();
  state.extractedVal = rewriter
                           .create(loc, opName, foldOperands,
                                   TypeRange{newResTy}, op->getAttrs())
                           ->getResult(0);
}

void ExtractAnalysis::visitOperand(Value operand, ExtractState &state,
                                   Operation *startOp, Location loc,
                                   PatternRewriter &rewriter) {
  auto *opInst = operand.getDefiningOp();
  if (!opInst)
    return getExtractedValueFrom(operand, state, loc, rewriter);

  auto *dialect = opInst->getDialect();
  if (mlir::isa<mlir::arith::ArithDialect>(dialect) ||
      mlir::isa<mlir::math::MathDialect>(dialect)) {
    return visitOperandFromOp(opInst, state, loc, rewriter);
  }

  // clang-format off
  return llvm::TypeSwitch<Operation *, void>(opInst)
      .Case<linalg::MapOp,
            linalg::FillOp,
            linalg_ext::MakeRangeOp,
            linalg::BroadcastOp,
            tensor::CollapseShapeOp,
            tensor::ExpandShapeOp>(
          [&](auto op) { return visitOperandFromOp(op, state, loc, rewriter); })
      .Default([&](Operation *op) {
        return getExtractedValueFrom(operand, state, loc, rewriter);
      });
  // clang-format on
}

template <typename OpTy>
LogicalResult ExtractAnalysis::rewriteExtractLikeOp(OpTy op,
                                                    PatternRewriter &rewriter) {
  static_assert(
      (std::is_same_v<OpTy, tensor::ExtractOp> ||
       std::is_same_v<
           OpTy,
           tensor::ExtractSliceOp>)&&"Only tensor.extract and "
                                     "tensor.extract_slice are supported yet.");

  // Only move rank-reduced tensor.extract_slice backward.
  if constexpr (std::is_same_v<OpTy, tensor::ExtractSliceOp>)
    if (!isOutputRankReduced(op))
      return failure();

  // Set as the flag for the original tensor.extract operation.
  ExtractState state{op};
  state.extractedVal = op.getResult();
  visitOperand(op.getOperand(0), state, op, op->getLoc(), rewriter);
  if (state.extractedVal == op.getResult())
    return failure();

  // Since tensor.extract_slice may erase unit dims, add a tensor.collapse_shape
  // when needed.
  if constexpr (std::is_same_v<OpTy, tensor::ExtractSliceOp>) {
    auto dstType = op.getResultType();
    state.extractedVal = reshapeToResultTypeByDropUnitDims(
        rewriter, op->getLoc(), dstType, state.extractedVal);
  }

  rewriter.replaceOp(op, state.extractedVal);

  return success();
}

} // anonymous namespace

/// Eliminate backward slice of value with no uses, which is used
/// to erase operations newly created to analyze whether scf.for
/// iter arguments can be replaced.
static void eliminateDeadExpressionsFrom(Value value,
                                         PatternRewriter &rewriter) {
  std::queue<Value> candidates({value});
  while (!candidates.empty()) {
    auto val = candidates.front();
    candidates.pop();

    if (mlir::isa<BlockArgument>(val))
      continue;

    auto *defOp = val.getDefiningOp();
    if (defOp->use_empty()) {
      for (auto opr : defOp->getOperands())
        candidates.push(opr);
      rewriter.eraseOp(defOp);
    }
  }
}

/// Get extract-like candidates, which meets the following constraints:
/// 1. scf.for is the parent op;
/// 2. use the iterIndex-th iter arguments as tensor operand;
/// 3. offsets/strides/sizes of the candidate are defined outside scf.for op.
template <typename OpTy>
static SetVector<Operation *> getCandidateExtractLikeOps(scf::ForOp forOp,
                                                         unsigned iterIndex) {
  Block *loopBody = &forOp.getRegion().front();
  auto arg = loopBody->getArgument(forOp.getNumInductionVars() + iterIndex);
  auto loopLikeOpInterface = cast<LoopLikeOpInterface>(forOp.getOperation());
  moveLoopInvariantCode(loopLikeOpInterface);

  SetVector<Operation *> candidates;
  for (auto *op : arg.getUsers()) {
    // Check that the for op is the parent of candidate.
    if (op->getParentOp() != forOp)
      continue;

    auto candidate = dyn_cast_or_null<OpTy>(op);
    if (!candidate)
      continue;

    auto srcToExtract = candidate.getOperand(0);
    if (srcToExtract != arg)
      continue;

    // Check that the offsets of tensor.extract are defined outside
    // current loop.
    if (llvm::any_of(candidate.getOperands().drop_front(), [&](Value value) {
          return !loopLikeOpInterface.isDefinedOutsideOfLoop(value);
        }))
      continue;

    if constexpr (std::is_same_v<OpTy, tensor::ExtractSliceOp>) {
      if (!isOutputRankReduced(candidate))
        continue;
    }

    candidates.insert(op);
  }

  return candidates;
}

/// Preconditions that ensure extract-like operations using the iterIndex iter
/// arguments can be extracted backward out of scf.for.
template <typename OpTy>
static LogicalResult extractIterArgPrecondition(scf::ForOp forOp,
                                                unsigned iterIndex,
                                                PatternRewriter &rewriter) {
  // Move loop invariant code ahead.
  auto loopLikeOpInterface =
      mlir::cast<LoopLikeOpInterface>(forOp.getOperation());
  moveLoopInvariantCode(loopLikeOpInterface);

  Block *loopBody = &forOp.getRegion().front();
  auto arg = loopBody->getArgument(forOp.getNumInductionVars() + iterIndex);

  auto extractCandidates = getCandidateExtractLikeOps<OpTy>(forOp, iterIndex);

  // Check that has extract operations to move backward.
  if (extractCandidates.empty())
    return failure();

  // Check whether the argument has def-use relations with other yield values.
  auto *yieldOp = &loopBody->back();
  std::queue<Value> qu({arg});
  while (!qu.empty()) {
    auto value = qu.front();
    qu.pop();
    for (auto *op : value.getUsers()) {
      // Since we would keep an equivalent computation logic for moved
      // extract-like operations, skip checking the result of extract-like
      // operations.
      if (extractCandidates.contains(op))
        continue;

      if (op == yieldOp && value != op->getOperand(iterIndex))
        return failure();

      for (auto res : op->getResults())
        qu.push(res);
    }
  }

  // Check that forward slice has no dependence on other operations in the for
  // opeation.
  SetVector<Operation *> forwardSlice;
  ForwardSliceOptions forwardSliceOptions;
  forwardSliceOptions.filter = [&forOp](Operation *op) {
    return !mlir::isa<OpTy>(op) && !mlir::isa<scf::YieldOp>(op) &&
           op->getParentOp() == forOp.getOperation() &&
           forOp->isProperAncestor(op);
  };
  getForwardSlice(arg, &forwardSlice, forwardSliceOptions);
  auto hasDependenceInsideLoop = [&](Operation *op) -> bool {
    for (auto operand : op->getOperands()) {
      if (loopLikeOpInterface.isDefinedOutsideOfLoop(operand))
        continue;
      if (arg == operand)
        continue;
      if (operand.getDefiningOp() &&
          forwardSlice.contains(operand.getDefiningOp()))
        continue;
      return true;
    }
    return false;
  };
  if (llvm::any_of(forwardSlice, hasDependenceInsideLoop))
    return failure();

  // Check whether new yield scalar operands can be computed by new iter
  // arguments.
  size_t extractNum = extractCandidates.size();
  SmallVector<ExtractState> states;
  states.reserve(extractNum);
  for (size_t index = 0; index < extractNum; ++index) {
    auto extractLikeOp = mlir::cast<OpTy>(extractCandidates[index]);
    auto operand = yieldOp->getOperand(iterIndex);
    ExtractState state{extractLikeOp};
    state.extractedVal = nullptr;
    states.push_back(state);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    ExtractAnalysis::visitOperand(operand, states[index], forOp, forOp.getLoc(),
                                  rewriter);
  }
  moveLoopInvariantCode(loopLikeOpInterface);
  for (size_t index = 0; index < extractNum; ++index) {
    std::queue<Value> qu({states[index].extractedVal});
    while (!qu.empty()) {
      Value currVal = qu.front();
      qu.pop();

      if (loopLikeOpInterface.isDefinedOutsideOfLoop(currVal))
        continue;

      if (mlir::isa<BlockArgument>(currVal) &&
          mlir::dyn_cast<BlockArgument>(currVal).getOwner()->getParentOp() ==
              forOp)
        return failure();

      auto *op = currVal.getDefiningOp();
      if (mlir::isa<OpTy>(op)) {
        if (op->getOperand(0) != loopBody->getArgument(
                                     iterIndex + forOp.getNumInductionVars()) ||
            !states[index].isSameExceptVal(mlir::cast<OpTy>(op))) {
          // Remove new inserted operations.
          eliminateDeadExpressionsFrom(states[index].extractedVal, rewriter);
          return failure();
        }
        continue;
      }

      for (auto opr : op->getOperands())
        qu.push(opr);
    }

    // Remove new inserted operations.
    eliminateDeadExpressionsFrom(states[index].extractedVal, rewriter);
  }

  return success();
}

/// Try to extract iter argument out of forOp region.
template <typename OpTy>
static LogicalResult tryExtractIterArgPrecondition(scf::ForOp op,
                                                   unsigned iterIndex,
                                                   PatternRewriter &rewriter) {
  // Check no users outside the loop operation.
  if (!op->getResult(iterIndex).use_empty())
    return failure();

  // Currently, it is impossible to determine whether the extraction out of the
  // for loop can be performed without modifying the IR. This leads to
  // modifications of the forOp's region regardless of whether the pattern to
  // extract out of the for loop can be executed, which will cause this pass to
  // get stuck. To circumvent this issue, a region (scf.execution_region) is
  // first created to make a copy of the for loop inside it. Then, analysis is
  // performed on the copied forOp. Regardless of whether it is successful, the
  // scf.execution_region is directly deleted in the end to avoid modifying the
  // original IR.
  // TODO: We need to find a way to analyze the extraction of operations
  // from the for loop without modifying the IR.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto exeOp = rewriter.create<scf::ExecuteRegionOp>(op->getLoc(), TypeRange{});
  rewriter.setInsertionPointToStart(&exeOp.getRegion().emplaceBlock());
  auto forOp = mlir::cast<scf::ForOp>(rewriter.clone(*op));
  auto ret = extractIterArgPrecondition<OpTy>(forOp, iterIndex, rewriter);
  rewriter.eraseOp(exeOp);
  return ret;
}

namespace {
template <typename OpTy>
struct ExtractRearrangementPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return ExtractAnalysis::rewriteExtractLikeOp(op, rewriter);
  }
};

/// Rewrite iteration arguments of scf.for operation with extracted value.
/// The rewrite process has the following constraits:
/// 1. Results of scf.for with the same index to the iteration arguments to be
/// rewritten has no uses.
/// 2. The iteration arguments to be rewritten only affect the yield operand
/// with the same index.
///
/// For example, consider the following examples:
/// Case1:
///
/// ``` mlir
/// func.func @extract_from_for_iter_args(%arg0: !llvm.ptr<f32, 1>,
///                                       %arg1: tensor<64x64xf32>,
///                                       %arg2: tensor<64x64xi64>,
///                                       %arg3: tensor<64x64xi64>)
///     -> tensor<64x64xf32> {
///   ...
///   %0:2 = scf.for %arg4 = %c0 to %c2 step %c1
///       iter_args(%arg5 = %arg1, %arg6 = %arg2)
///       -> (tensor<64x64xf32>, tensor<64x64xi64>) {
///     %1 = tensor.extract %arg6[%c0, %c0] : tensor<64x64xi64>
///     %2 = arith.index_cast %1 : i64 to index
///     %3 = aux.view %arg0[%2] : !llvm.ptr<f32, 1> to tensor<64x?xf32>
///     %4 = tensor.extract_slice %3[%2, %2] [64, 64] [1, 1] :
///         tensor<64x?xf32> to tensor<64x64xf32>
///     %5 = linalg.init_tensor [64, 64] : tensor<64x64xi64>
///     %mapped = linalg.map { arith.addi }
///         ins(%arg6, %arg3 : tensor<64x64xi64>, tensor<64x64xi64>)
///         outs(%5 : tensor<64x64xi64>)
///     scf.yield %4, %mapped : tensor<64x64xf32>, tensor<64x64xi64>
///   }
///   ...
/// }
/// ```
///
/// After running, we get the expected:
///
/// ``` mlir
/// func.func @extract_from_for_iter_args(%arg0: !llvm.ptr<f32, 1>,
///                                       %arg1: tensor<64x64xf32>,
///                                       %arg2: tensor<64x64xi64>,
///                                       %arg3: tensor<64x64xi64>)
///     -> tensor<64x64xf32> {
///   ...
///   %0 = tensor.extract %arg2[%c0, %c0] : tensor<64x64xi64>
///   %1 = tensor.extract %arg3[%c0, %c0] : tensor<64x64xi64>
///   %2:3 = scf.for %arg4 = %c0 to %c2 step %c1
///      iter_args(%arg5 = %arg1, %arg6 = %arg2, %arg7 = %0)
///      -> (tensor<64x64xf32>, tensor<64x64xi64>, i64) {
///     %3 = arith.index_cast %arg7 : i64 to index
///     %4 = aux.view %arg0[%3] : !llvm.ptr<f32, 1> to tensor<64x?xf32>
///     %5 = tensor.extract_slice %4[%3, %3] [64, 64] [1, 1] :
///         tensor<64x?xf32> to tensor<64x64xf32>
///     %6 = arith.addi %arg7, %1 : i64
///     scf.yield %4, %arg6, %6 : tensor<64x64xf32>, tensor<64x64xi64>, i64
///   }
///   ...
/// }
/// ```
///
/// Case2:
///
/// ``` mlir
/// func.func @extract_slice_from_for_iter_args(%arg0: i64,
///                                             %arg1: tensor<64x64xf32>,
///                                             %arg2: tensor<32x32xf32>)
///     -> tensor<32x32xf32> {
///   %cst = arith.constant 2.000000e+00 : f32
///   %0:2 = scf.for %arg3 = %c0 to %c2 step %c1
///      iter_args(%arg4 = %arg1, %arg5 = %arg2)
///      -> (tensor<64x64xf32>, tensor<32x32xf32>) {
///     %extracted_slice = tensor.extract_slice %arg4[0, 0] [32, 32] [1, 1] :
///         tensor<64x64xf32> to tensor<32x32xf32>
///     %mapped = arith.addf %extracted_slice, %arg5 : tensor<32x32xf32>
///     %mapped_0 = arith.addf %arg4, %cst : tensor<64x64xf32>
///     scf.yield %mapped_0, %mapped : tensor<64x64xf32>, tensor<32x32xf32>
///   }
///   return %0#1 : tensor<32x32xf32>
/// }
/// ```
///
/// After running, we get the expected:
///
/// ``` mlir
/// func.func @extract_slice_from_for_iter_args(%arg0: i64,
///                                             %arg1: tensor<64x64xf32>,
///                                             %arg2: tensor<32x32xf32>)
///     -> tensor<32x32xf32> {
///   %extracted_slice = tensor.extract_slice %arg1[0, 0] [32, 32] [1, 1] :
///       tensor<64x64xf32> to tensor<32x32xf32>
///   %0:2 = scf.for %arg3 = %c0 to %c2 step %c1
///      iter_args(%arg4 = %extracted_slice, %arg5 = %arg2)
///      -> (tensor<32x32xf32>, tensor<32x32xf32>) {
///     %mapped = arith.addf %arg4, %arg5 : tensor<32x32xf32>
///     %mapped_0 = arith.addf %arg4, %cst : tensor<32x32xf32>
///     scf.yield %mapped_0, %mapped : tensor<32x32xf32>, tensor<32x32xf32>
///   }
///   return %0#1 : tensor<32x32xf32>
/// }
/// ```
///
/// Note that the unused iteration argument can be erased by canonicalizer.
template <typename OpTy>
struct SCFRearrangementPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    LogicalResult ret = failure();
    for (unsigned iterOperandNumber = 0, e = forOp.getNumRegionIterArgs();
         iterOperandNumber < e; ++iterOperandNumber) {
      if (failed(tryExtractIterArgPrecondition<OpTy>(forOp, iterOperandNumber,
                                                     rewriter)))
        continue;

      Block *oldLoopBody = &forOp.getRegion().front();

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(forOp);

      auto createExtractLikeOpWithNewSrc = [&](Value src,
                                               Operation *op) -> Value {
        SmallVector<Value> operands;
        operands.push_back(src);
        auto posOperands = op->getOperands().drop_front();
        operands.append(posOperands.begin(), posOperands.end());
        SmallVector<Type> resultTypes{op->getResult(0).getType()};
        OperationState opState(op->getLoc(), op->getName(), operands,
                               resultTypes, op->getAttrs());
        return rewriter.create(opState)->getResult(0);
      };

      // Move extract-like operations ahead of scf.for.
      auto extractLikeOpsToMove =
          getCandidateExtractLikeOps<OpTy>(forOp, iterOperandNumber);
      SmallVector<Value> newExtractLikeOps;
      for (auto *extract : extractLikeOpsToMove) {
        auto src = forOp.getInitArgsMutable()[iterOperandNumber].get();
        newExtractLikeOps.push_back(
            createExtractLikeOpWithNewSrc(src, extract));
      }

      // Add new operands to scf.for.
      SmallVector<Value> newIterArgs = forOp.getInitArgs();
      newIterArgs.append(newExtractLikeOps);

      auto newForOp = rewriter.create<scf::ForOp>(
          forOp->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newIterArgs);

      // Update scf.for operands.
      Block *loopBody = &newForOp.getRegion().front();
      rewriter.setInsertionPointToStart(loopBody);
      IRMapping bvm;
      for (auto [oldArg, newArg] :
           zip(oldLoopBody->getArguments(), loopBody->getArguments()))
        bvm.map(oldArg, newArg);
      for (const auto &en : enumerate(extractLikeOpsToMove))
        bvm.map(
            en.value()->getResult(0),
            loopBody->getArgument(oldLoopBody->getNumArguments() + en.index()));

      // Clone operations from old loop body to the new one.
      llvm::MapVector<Operation *, Operation *> oldAndNewOpMap;
      for (auto &op : *oldLoopBody) {
        if (extractLikeOpsToMove.contains(&op) || mlir::isa<scf::YieldOp>(&op))
          continue;
        auto *newOp = rewriter.clone(op, bvm);
        oldAndNewOpMap[&op] = newOp;
      }

      SmallVector<Value> newYieldValues;
      for (auto yieldValue : oldLoopBody->back().getOperands()) {
        auto newYieldValue = bvm.lookup(yieldValue);
        newYieldValues.push_back(newYieldValue);
      }

      // Update init args according to forward slice operations.
      Value oldYieldValue = oldLoopBody->back().getOperand(iterOperandNumber);
      auto newYieldValue = bvm.lookup(oldYieldValue);
      SmallVector<Value> updatedYieldValues;
      for (const auto &en : enumerate(extractLikeOpsToMove)) {
        // Re-extract yield scarlar values from yield values.
        auto updatedYieldValue =
            createExtractLikeOpWithNewSrc(newYieldValue, en.value());
        updatedYieldValues.push_back(updatedYieldValue);
      }
      newYieldValues.append(updatedYieldValues);

      newYieldValues[iterOperandNumber] = loopBody->getArgument(
          forOp.getNumInductionVars() + iterOperandNumber);
      rewriter.create<scf::YieldOp>(oldLoopBody->back().getLoc(),
                                    newYieldValues);

      // Try to move new inserted extract-like operations forward.
      for (auto value : updatedYieldValues)
        (void)ExtractAnalysis::rewriteExtractLikeOp(value.getDefiningOp<OpTy>(),
                                                    rewriter);

      // Replace the users of extract-like operations with new created block
      // argument.
      SmallVector<std::pair<Operation *, Value>> replacePairs;
      for (Operation *op :
           loopBody
               ->getArgument(iterOperandNumber + newForOp.getNumInductionVars())
               .getUsers()) {
        if (!mlir::isa<OpTy>(op))
          continue;
        Value substitute;
        ExtractState targetState{mlir::cast<OpTy>(op)};
        for (const auto &en : llvm::enumerate(extractLikeOpsToMove)) {
          if (!targetState.isSameExceptVal(mlir::cast<OpTy>(en.value())))
            continue;
          substitute =
              loopBody->getArgument(en.index() + loopBody->getNumArguments() -
                                    extractLikeOpsToMove.size());
          break;
        }

        if (!substitute)
          return rewriter.notifyMatchFailure(forOp->getLoc(),
                                             "Fail to find new iter argument.");
        replacePairs.push_back({op, substitute});
      }

      for (auto p : replacePairs) {
        rewriter.setInsertionPointToStart(loopBody);
        if (auto extractSliceOp =
                mlir::dyn_cast<tensor::ExtractSliceOp>(p.first)) {
          auto resultType = extractSliceOp.getResultType();
          p.second = expandShapeToResultTypeByAddUnitDims(
              rewriter, forOp.getLoc(), resultType, p.second);
        }
        rewriter.replaceOp(p.first, p.second);
      }

      rewriter.replaceOp(
          forOp, newForOp->getResults().drop_back(extractLikeOpsToMove.size()));

      // Update forOp for next iter argument rewritten.
      forOp = newForOp;

      // Note that this pattern scans all init arguments and tries to move all
      // extract-like operations which use these init arguments as tensor
      // operand. And update rewrite status to succeed if at least one init
      // argument can be rewritten.
      ret = success();
    }
    return ret;
  }
};

struct ExtractLikeMoveBackwardPass
    : public triton::ExtractLikeMoveBackwardPassBase<
          ExtractLikeMoveBackwardPass> {
  explicit ExtractLikeMoveBackwardPass() = default;

  void runOnOperation() override {
    auto *context = &getContext();
    GreedyRewriteConfig config;
    bool changed = false;

    // FIXME: Starting from LLVM19, during conversion, if the ParentOp of
    // an Op is also in the same conversion pattern, accessing the ParentOp from
    // within the Op may be an invalid behavior.
    do {
      RewritePatternSet extractPatterns(context);
      extractPatterns.add<ExtractRearrangementPattern<tensor::ExtractOp>,
                          ExtractRearrangementPattern<tensor::ExtractSliceOp>>(
          context);

      RewritePatternSet scfPatterns(context);
      scfPatterns.add<SCFRearrangementPattern<tensor::ExtractOp>,
                      SCFRearrangementPattern<tensor::ExtractSliceOp>>(context);

      bool extractChanged = false;
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(extractPatterns),
                                              config, &extractChanged)))
        return signalPassFailure();

      bool scfChanged = false;
      if (failed(applyPatternsAndFoldGreedily(
              getOperation(), std::move(scfPatterns), config, &scfChanged)))
        return signalPassFailure();

      changed = extractChanged && scfChanged;
    } while (changed);
  }
};
} // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::triton::createExtractLikeMoveBackwardPass() {
  return std::make_unique<ExtractLikeMoveBackwardPass>();
}
