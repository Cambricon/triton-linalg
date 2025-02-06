//===- MaskTracker.cpp - Trace the mask pattern -----------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <variant>

#include "triton-linalg/Dialect/Triton/Utils/MaskTracker.h"
#include "triton-linalg/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mask-tracker"

using namespace mlir;
using namespace mlir::triton;
namespace mlir {
class Operation;
} // namespace mlir

/// Figure if the shape's indices dimension match `<1xNx1>`.
static bool isSingleNonOneSizeIndShape(ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> indices) {
  assert(indices.size() <= shape.size());
  int64_t numNonOneSize =
      std::count_if(indices.begin(), indices.end(),
                    [&](int64_t dim) { return shape[dim] != 1; });
  return numNonOneSize == 1;
}

/// Figure if the shape's indices dimension match `<1x1x1>`.
static bool isAllOneSizeIndShape(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> indices) {
  assert(indices.size() <= shape.size());
  int64_t numOfOnes =
      std::count_if(indices.begin(), indices.end(),
                    [&](int64_t dim) { return shape[dim] == 1; });
  return numOfOnes == indices.size();
}

/// Find which reassociative axis is one size, for example:
///  1.1 tensor.expand_shape <N> -> <1xNx1> : [0, 2]
///  1.2 tensor.expand_shape <1> -> <1x1x1> : [1, 2]
///  1.3 tensor.expand_shape <MxN> -> <Mx1xNx1> : [0, 2]
///  1.4 tensor.expand_shape <MxN> -> <M1xM2xNx1> : null
///  2.1.tensor.collapse_shape <1xNx1> -> <N> : [0, 2]
///  2.2 tensor.collapse_shape <1x1x1> -> <1> : [1, 2]
///  2.3 tensor.collapse_shape <Mx1xNx1> -> <MxN> : [0, 2]
///  2.4 tensor.collapse_shape <M1xM2xNx1> -> <MxN> : null
static std::optional<SmallVector<int64_t>>
getOneSizeAxesOfReassociativeReshapeOp(
    Operation *op, std::optional<int64_t> skippedAxis = std::nullopt) {
  SmallVector<int64_t, 4> axes;
  SmallVector<ReassociationIndices, 4> reassociationIndices;
  ArrayRef<int64_t> expandedShape;
  if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    expandedShape = expandShapeOp.getResultType().getShape();
    reassociationIndices = expandShapeOp.getReassociationIndices();
  } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    expandedShape = collapseShapeOp.getSrcType().getShape();
    reassociationIndices = collapseShapeOp.getReassociationIndices();
  } else {
    return std::nullopt;
  }
  for (const auto &indices : reassociationIndices) {
    // If any indice shape does not match <1xNx1> or <1x1x1>, failed as the
    // mask tracker not support analysis `<M> <-> <M1xM2>`.
    if (!isSingleNonOneSizeIndShape(expandedShape, indices) &&
        !isAllOneSizeIndShape(expandedShape, indices))
      return std::nullopt;
    if (indices.size() == 1)
      continue;
    // If skipped axis not specified, keep the first dim of indice for pattern
    // `<1> <-> <1x1x1>`.
    if (skippedAxis == std::nullopt &&
        isAllOneSizeIndShape(expandedShape, indices)) {
      skippedAxis = indices.front();
    }
    for (int64_t indice : indices) {
      if (skippedAxis.has_value() && indice == skippedAxis.value())
        continue;
      if (expandedShape[indice] == 1) {
        axes.push_back(indice);
      }
    }
  }
  return axes;
}

namespace {
inline raw_ostream &operator<<(raw_ostream &os, ArrayRef<OpFoldResult> vec) {
  os << "[";
  if (!vec.empty()) {
    for (auto val : llvm::drop_end(vec, 1)) {
      os << val << ", ";
    }
    os << vec.back();
  }
  os << "]";
  return os;
}

struct Scalar;
struct SimpleRange;
struct Mask;

inline raw_ostream &operator<<(raw_ostream &os, const Scalar &s);
inline raw_ostream &operator<<(raw_ostream &os, const SimpleRange &s);
inline raw_ostream &operator<<(raw_ostream &os, const Mask &s);

/// Base class for intermediate results information in mask tracking process.
struct StateBase {
  StateBase() = default;
  StateBase(int64_t rank) : dims(rank, nullptr) {}
  int64_t getRank() const { return dims.size(); }
  /// Describing the length of data that is continuous for each dimension.
  SmallVector<OpFoldResult> dims;
};

/// Scalar data.
struct Scalar : public StateBase {
  void dump() const { llvm::errs() << *this << "\n"; }
  OpFoldResult scalar;
};

/// Describe a int range.
/// Closed lower bound and open upper bound range([start, end)).
struct SimpleRange : public StateBase {
  using StateBase::StateBase;
  bool isTrackingAxis() const { return axis != kUnknownAxis; }
  bool isAxisTrackingInvalid() const { return isTrackingAxis() && !dims[axis]; }
  void dump() const { llvm::errs() << *this << "\n"; }
  int64_t axis{kUnknownAxis};
  /// Range lower bound(strat) and upper bound(end)
  OpFoldResult start, end;

  /// Init axis value, invalid axis value.
  constexpr static int32_t kUnknownAxis = -1;
};

/// Describe the continuity information of a bool tensor.
struct Mask : public StateBase {
  Mask(int64_t rank)
      : StateBase(rank), maskStarts(rank, nullptr), maskEnds(rank, nullptr) {}
  bool isAxisTrackingInvalid(int64_t axis) const { return !dims[axis]; }
  // Set mask dim range is [0, dim).
  // When dim is nullptr, we will set start/end/dim to nullptr.
  void setFullDimensionMask(OpBuilder &b, OpFoldResult dim, int64_t dimNum) {
    dims[dimNum] = dim;
    maskEnds[dimNum] = dim;
    if (dim) {
      maskStarts[dimNum] = b.getIndexAttr(0);
    }
  }
  void dump() const { llvm::errs() << *this << "\n"; }
  /// Describe which interval([maskStart, maskEnd)) has the same bool values.
  SmallVector<OpFoldResult> maskStarts, maskEnds;
};

inline raw_ostream &operator<<(raw_ostream &os, const Scalar &s) {
  os << "Scalar { value : " << s.scalar << "; dims : " << s.dims << "; }";
  return os;
}

inline raw_ostream &operator<<(raw_ostream &os, const SimpleRange &s) {
  os << "SimpleRange { axis : " << s.axis << "; start : " << s.start
     << "; end : " << s.end << "; dims : " << s.dims << "; }";
  return os;
}

inline raw_ostream &operator<<(raw_ostream &os, const Mask &s) {
  os << "Mask { starts : [" << s.maskStarts << "; ends : " << s.maskEnds
     << "; dims : " << s.dims << "; }";
  return os;
}

using Result = std::variant<Scalar, SimpleRange, Mask>;
inline raw_ostream &operator<<(raw_ostream &os, const Result &s) {
  std::visit([&os](const auto &val) { os << val; }, s);
  return os;
}

/// A visitor(std::visit functor) used to wrapper calculations between different
/// of results.
struct VisitorBase {
public:
  VisitorBase(Location loc, RewriterBase &rewriter)
      : loc(loc), rewriter(rewriter) {}

protected:
  Location loc;
  RewriterBase &rewriter;
};

struct BinaryVisitor : public VisitorBase {
  using computeFnTy = llvm::function_ref<OpFoldResult(
      OpFoldResult, OpFoldResult, Location, OpBuilder &)>;
  BinaryVisitor(Location loc, RewriterBase &rewriter, computeFnTy fn)
      : VisitorBase(loc, rewriter), computeFn(fn) {}
  template <typename T1, typename T2>
  FailureOr<Result> operator()(const T1 &t1, const T2 &t2) {
    if constexpr (std::is_base_of_v<SimpleRange, T1> &&
                  std::is_same_v<Scalar, T2>) {
      auto ret = t1;
      computeRangeAndScalar(ret, t2);
      return Result(ret);
    }
    if constexpr (std::is_base_of_v<SimpleRange, T2> &&
                  std::is_same_v<Scalar, T1>) {
      auto ret = t2;
      computeRangeAndScalar(ret, t1);
      return Result(ret);
    }
    if constexpr (std::is_same_v<Scalar, T1> && std::is_same_v<Scalar, T2>) {
      auto ret = t1;
      computeScalarAndScalar(ret, t2);
      return Result(ret);
    }
    return rewriter.notifyMatchFailure(
        loc, "Currently, computing tow SimpleRange typesis not supported");
  }

private:
  computeFnTy computeFn = nullptr;
  void computeRangeAndScalar(SimpleRange &ret, const Scalar &scalar) {
    if (!ret.start)
      return;
    ret.start = computeFn(ret.start, scalar.scalar, loc, rewriter);
    ret.end = computeFn(ret.end, scalar.scalar, loc, rewriter);
  }
  void computeScalarAndScalar(Scalar &ret, const Scalar &scalar) {
    ret.scalar = computeFn(ret.scalar, scalar.scalar, loc, rewriter);
  }
};

struct MergeVisitor : public VisitorBase {
  using VisitorBase::VisitorBase;
  template <typename T1, typename T2>
  FailureOr<Result> operator()(const T1 &lhs, const T2 &rhs) {
    if constexpr (std::is_same_v<Mask, T1> && std::is_same_v<Mask, T2>) {
      return mergeImpl(lhs, rhs);
    }
    return failure();
  }

private:
  FailureOr<Result> mergeImpl(const Mask &lhs, const Mask &rhs) {
    if (lhs.getRank() != rhs.getRank()) {
      return rewriter.notifyMatchFailure(
          loc, "Unexpected case where lhs and rhs have different ranks");
    }
    auto rank = lhs.getRank();
    Mask ret(rank);
    for (uint32_t i = 0; i < rank; i++) {
      auto lhsStart = lhs.maskStarts[i];
      auto rhsStart = rhs.maskStarts[i];
      auto lhsEnd = lhs.maskEnds[i];
      auto rhsEnd = rhs.maskEnds[i];
      // Unknown axis mask state.
      if (lhs.isAxisTrackingInvalid(i) || rhs.isAxisTrackingInvalid(i) ||
          !lhsStart || !rhsStart) {
        continue;
      }
      ret.maskStarts[i] = maxOFRs(lhsStart, rhsStart, loc, rewriter);
      ret.maskEnds[i] = minOFRs(lhsEnd, rhsEnd, loc, rewriter);
      ret.dims[i] = subOFRs(ret.maskEnds[i], ret.maskStarts[i], loc, rewriter);
      // We should ensure dim is greater than 0.
      ret.dims[i] =
          maxOFRs(ret.dims[i], rewriter.getIndexAttr(0), loc, rewriter);
    }
    return Result(ret);
  }
};

struct CmpVisitor : public VisitorBase {
  using VisitorBase::VisitorBase;
  CmpVisitor(Location loc, RewriterBase &rewriter, arith::CmpIPredicate cmpTy)
      : VisitorBase(loc, rewriter), cmpTy(cmpTy) {}
  template <typename T1, typename T2>
  FailureOr<Result> operator()(const T1 &lhs, const T2 &rhs) {
    // Compare range and scalar.
    if constexpr (std::is_same_v<SimpleRange, T1> &&
                  std::is_same_v<Scalar, T2>) {
      return compareSimpleRange(lhs, rhs, cmpTy);
    }
    // Compare scalar and range.
    if constexpr (std::is_same_v<Scalar, T1> &&
                  std::is_same_v<SimpleRange, T2>) {
      // The function compareSimpleRange handles comparisons between
      // a range and a scalar. When the input is in the form
      // of (scalar, range), it is necessary to swap the order of the
      // arguments and map the comparison operation to its equivalent operation.
      return compareSimpleRange(rhs, lhs, reversePredicate(cmpTy));
    }
    return rewriter.notifyMatchFailure(loc, "Unsupported cmpi scenario");
  }

private:
  arith::CmpIPredicate cmpTy;

  arith::CmpIPredicate reversePredicate(mlir::arith::CmpIPredicate cmpTy) {
    /*
     *  (range < scalar) <=> (scalar > range)
     *  (range > scalar) <=> (scalar < range)
     *  (range <= scale) <=> (scalar >= range)
     *  (range >= scalar) <=> (scalar <= range)
     *  (range == scalar) <=> (scalar == range)
     */
    static const llvm::DenseMap<arith::CmpIPredicate, arith::CmpIPredicate>
        map = {
            {arith::CmpIPredicate::slt, arith::CmpIPredicate::sgt},
            {arith::CmpIPredicate::sgt, arith::CmpIPredicate::slt},
            {arith::CmpIPredicate::sle, arith::CmpIPredicate::sge},
            {arith::CmpIPredicate::sge, arith::CmpIPredicate::sle},
            {arith::CmpIPredicate::eq, arith::CmpIPredicate::eq},
        };
    auto it = map.find(cmpTy);
    assert(it != map.end());
    return it->second;
  }
  FailureOr<Result> compareSimpleRange(const SimpleRange &lhs,
                                       const Scalar &rhs,
                                       mlir::arith::CmpIPredicate cmpTy) {
    auto rank = lhs.getRank();
    Mask ret(rank);
    // We think cmpi lhs only carry one dim infomation,
    // so we fill axis start/end value and other dim use nullptr as a
    // placeholder.
    for (int32_t i = 0; i < rank; i++) {
      if (i == lhs.axis && !lhs.isAxisTrackingInvalid()) {
        OpFoldResult newDim;
        switch (cmpTy) {
        case arith::CmpIPredicate::slt: {
          newDim = cmpSlt(ret, lhs, rhs.scalar, i);
          break;
        }
        case arith::CmpIPredicate::sle: {
          auto openedUpperBound =
              addOFRs(rhs.scalar, rewriter.getIndexAttr(1), loc, rewriter);
          // The value of `rhs.scalar` might be exactly `INT64_MAX`.
          // We need to prevent overflow after adding one.
          openedUpperBound =
              maxOFRs(openedUpperBound, rhs.scalar, loc, rewriter);
          newDim = cmpSlt(ret, lhs, openedUpperBound, i);
          break;
        }
        case arith::CmpIPredicate::sgt: {
          auto closedLowerBound =
              addOFRs(rhs.scalar, rewriter.getIndexAttr(1), loc, rewriter);
          closedLowerBound =
              maxOFRs(closedLowerBound, rhs.scalar, loc, rewriter);
          newDim = cmpSgt(ret, lhs, closedLowerBound, i);
          break;
        }
        case arith::CmpIPredicate::sge: {
          newDim = cmpSgt(ret, lhs, rhs.scalar, i);
          break;
        }
        case arith::CmpIPredicate::eq: {
          newDim = cmpEq(ret, lhs, rhs.scalar, i);
          break;
        }
        default: {
          return rewriter.notifyMatchFailure(loc, "Unsupport compare type");
        }
        }
        ret.dims[i] = newDim;
      } else {
        ret.setFullDimensionMask(rewriter, lhs.dims[i], i);
      }
    }
    return Result(ret);
  }
  OpFoldResult cmpSlt(Mask &ret, const SimpleRange &lhs, OpFoldResult openedUb,
                      int32_t i) {
    OpFoldResult absoluteStart, absoluteEnd, newDim;
    absoluteStart = lhs.start;
    auto limitUb = maxOFRs(openedUb, lhs.start, loc, rewriter);
    absoluteEnd = minOFRs(lhs.end, limitUb, loc, rewriter);
    newDim = subOFRs(absoluteEnd, absoluteStart, loc, rewriter);
    ret.maskStarts[i] = rewriter.getIndexAttr(0);
    ret.maskEnds[i] = newDim;
    return newDim;
  }
  OpFoldResult cmpSgt(Mask &ret, const SimpleRange &lhs, OpFoldResult closedLb,
                      int32_t i) {
    OpFoldResult absoluteStart, absoluteEnd, newDim;
    // We should use closed lower bound, but sgt result is open lower
    // bound. We need to add 1 to max result.
    auto limitLb = minOFRs(lhs.end, closedLb, loc, rewriter);
    absoluteStart = maxOFRs(lhs.start, limitLb, loc, rewriter);
    absoluteEnd = lhs.end;
    newDim = subOFRs(absoluteEnd, absoluteStart, loc, rewriter);
    ret.maskStarts[i] = subOFRs(absoluteStart, lhs.start, loc, rewriter);
    ret.maskEnds[i] = addOFRs(ret.maskStarts[i], newDim, loc, rewriter);
    return newDim;
  }
  OpFoldResult cmpEq(Mask &ret, const SimpleRange &lhs, OpFoldResult closedLb,
                     int32_t i) {
    OpFoldResult absoluteStart, absoluteEnd, newDim;
    // Step1: get absolute start according to `sge` implement.
    auto limitLb = minOFRs(lhs.end, closedLb, loc, rewriter);
    absoluteStart = maxOFRs(lhs.start, limitLb, loc, rewriter);
    // Step2: get absolute end according to `sle` implement.
    auto openedUb = addOFRs(closedLb, rewriter.getIndexAttr(1), loc, rewriter);
    auto limitUb = maxOFRs(openedUb, lhs.start, loc, rewriter);
    absoluteEnd = minOFRs(lhs.end, limitUb, loc, rewriter);
    // Step3: get `eq` range according to `eq` <=> `sge && sle`.
    newDim = subOFRs(absoluteEnd, absoluteStart, loc, rewriter);
    // Step4: set dim's upper bound to 1 as `range eq scalar` always be 0 or 1.
    newDim = minOFRs(newDim, rewriter.getIndexAttr(1), loc, rewriter);
    ret.maskStarts[i] = subOFRs(absoluteStart, lhs.start, loc, rewriter);
    ret.maskEnds[i] = addOFRs(ret.maskStarts[i], newDim, loc, rewriter);
    return newDim;
  }
};

struct BroadcastVisitor : public VisitorBase {
  BroadcastVisitor(Location loc, RewriterBase &rewriter,
                   ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape)
      : VisitorBase(loc, rewriter), srcShape(srcShape), dstShape(dstShape) {}
  template <typename T> LogicalResult operator()(T &self) {
    if constexpr (std::is_base_of_v<SimpleRange, T>) {
      return broadcastSimpleRange(self, srcShape, dstShape);
    }
    if constexpr (std::is_same_v<Mask, T>) {
      return broadcastMask(self, srcShape, dstShape);
    }
    if constexpr (std::is_same_v<Scalar, T>) {
      return broadcastScalar(self, srcShape, dstShape);
    }
    return rewriter.notifyMatchFailure(loc, "Unsupported broadcast");
  }

private:
  ArrayRef<int64_t> srcShape, dstShape;
  LogicalResult broadcastMask(Mask &self, ArrayRef<int64_t> srcShape,
                              ArrayRef<int64_t> dstShape) {
    // The behavior of broadcasting masks is quite unique. When there is an axis
    // with a length not equal to 1, the result of broadcasting the mask is
    // unaffected by axes with a length of 1. That is, the axes being
    // broadcasted do not influence the outcome of the mask. Therefore, in cases
    // where there are axes with lengths not equal to 1, we can forcefully
    // refresh axes with a length of 1 to be full size, regardless of whether
    // they were successfully broadcasted or not. However, when there are no
    // axes with lengths other than 1, the mask information for each axis is
    // intertwined, and we cannot forcefully refresh in this scenario. An axis
    // that fails to broadcast remains a failure, and an axis that succeeds
    // remains a success.
    //
    // Example:
    //   case1:
    //   %1 = load %ptr -> (tensor<8x32x1xi32>)
    //   %2 = cmp %1, %c32 -> (tensor<8x32x1xi32>)
    //   %3 = broadcast %2 : (tensor<8x32x1xi1>) -> ((tensor<8x32x128xi1>))
    //   mask result is : start {?, ?, 0}; dims {?, ?, 128}
    //   -------
    //   case2:
    //   %1 = load %ptr -> (tensor<1x1xi32>)
    //   %2 = cmp %1, %c32 -> (tensor<1x1xi32>) False
    //   %3 = broadcast %2 : (tensor<1x1xi1>) -> ((tensor<1x128xi1>))
    //   %4 = broadcast %3 : (tensor<1x128xi1>) -> ((tensor<32x128xi1>))
    //   mask result is : start {0, ?}; dims {32, ?}
    bool isOneElement =
        !llvm::any_of(srcShape, [](auto val) { return val != 1; });
    auto setMask = [&self, this, isOneElement](size_t idx, int64_t dim) {
      // When src is only one element and current axis is failed,
      // we need to set current axis is failed.
      if (isOneElement && !self.maskStarts[idx]) {
        self.dims[idx] = nullptr;
        self.maskStarts[idx] = nullptr;
        self.maskEnds[idx] = nullptr;
      } else {
        if (!self.maskStarts[idx])
          self.maskStarts[idx] = rewriter.getIndexAttr(0);
        self.maskEnds[idx] = rewriter.getIndexAttr(dim);
      }
    };
    return broadcastImpl(self, srcShape, dstShape, nullptr, setMask);
  }

  LogicalResult broadcastSimpleRange(SimpleRange &self,
                                     ArrayRef<int64_t> srcShape,
                                     ArrayRef<int64_t> dstShape) {
    auto checkAxis = [&self, this](size_t idx) {
      if (self.isTrackingAxis() && self.axis == idx)
        return rewriter.notifyMatchFailure(loc,
                                           "Unsupport range axis broadcast.");
      return success();
    };
    return broadcastImpl(self, srcShape, dstShape, checkAxis);
  }

  LogicalResult broadcastScalar(Scalar &self, ArrayRef<int64_t> srcShape,
                                ArrayRef<int64_t> dstShape) {
    return broadcastImpl(self, srcShape, dstShape);
  }

  LogicalResult broadcastImpl(
      StateBase &self, ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
      llvm::function_ref<LogicalResult(size_t)> checkBroadcastAxis = nullptr,
      llvm::function_ref<void(int64_t, int64_t)> setValue = nullptr) {
    for (int64_t i = 0; i < static_cast<int64_t>(srcShape.size()); i++) {
      if (srcShape[i] < dstShape[i]) {
        if (checkBroadcastAxis && failed(checkBroadcastAxis(i)))
          return failure();
        self.dims[i] = rewriter.getIndexAttr(dstShape[i]);
        if (setValue)
          setValue(i, dstShape[i]);
      }
    }
    return success();
  }
};

struct SplatVisitor : public VisitorBase {
  SplatVisitor(Location loc, RewriterBase &rewriter, ArrayRef<int64_t> dstShape,
               Type elemType)
      : VisitorBase(loc, rewriter), dstShape(dstShape), elemType{elemType} {}
  template <typename T> FailureOr<Result> operator()(T &self) {
    if constexpr (std::is_same_v<Scalar, T>) {
      return splatScalar(self, dstShape);
    }
    llvm_unreachable("Unsupported splat");
  }

private:
  ArrayRef<int64_t> dstShape;
  Type elemType;
  FailureOr<Result> splatScalar(Scalar &self, ArrayRef<int64_t> dstShape) {
    assert(self.dims.empty() && "splat only support 0 rank scalar input");

    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (intType.getWidth() == 1) {
        // For type i1, we can consider it as a mask. The src of splatop
        // is a scalar, which can be used to mark the length of dimensions with
        // shape=1, i.e., cast_to_int(mask). If there are multiple dimensions
        // with shape=1, it defaults to marking the first dimension.
        Mask ret(dstShape.size());
        bool succeed = false;

        for (auto e : llvm::enumerate(dstShape)) {
          auto idx = e.index();
          ret.maskStarts[idx] = rewriter.getIndexAttr(0);

          if (!succeed && e.value() == 1) {
            ret.maskEnds[idx] = self.scalar;
            succeed = true;
          } else {
            ret.maskEnds[idx] = rewriter.getIndexAttr(e.value());
          }
          ret.dims[idx] = ret.maskEnds[idx];
        }
        if (succeed) {
          return Result(ret);
        } else {
          return failure();
        }
      }
    }

    for (auto s : dstShape)
      self.dims.push_back(rewriter.getIndexAttr(s));
    return Result(self);
  }
};

struct ExpandDimVisitor : public VisitorBase {
  ExpandDimVisitor(Location loc, RewriterBase &rewriter, int64_t axis)
      : VisitorBase(loc, rewriter), axis(axis) {}
  template <typename T> LogicalResult operator()(T &self) {
    if constexpr (std::is_base_of_v<SimpleRange, T>) {
      return expandDimRange(self, axis);
    } else if constexpr (std::is_same_v<Mask, T>) {
      return expandDimMask(self, axis);
    } else if constexpr (std::is_same_v<Scalar, T>) {
      return expandDimScalar(self, axis);
    }
    return rewriter.notifyMatchFailure(loc, "Unsupported expand_dim");
  }

private:
  int64_t axis;
  LogicalResult expandDimScalar(Scalar &self, int64_t axis) {
    self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
    return success();
  }
  LogicalResult expandDimRange(SimpleRange &self, int64_t axis) {
    self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
    if (self.isTrackingAxis() && axis <= self.axis)
      self.axis += 1;
    return success();
  }
  LogicalResult expandDimMask(Mask &self, int64_t axis) {
    self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
    self.maskStarts.insert(self.maskStarts.begin() + axis,
                           rewriter.getIndexAttr(0));
    self.maskEnds.insert(self.maskEnds.begin() + axis,
                         rewriter.getIndexAttr(1));
    return success();
  }
};

struct ExpandShapeVisitor : public VisitorBase {
  ExpandShapeVisitor(Location loc, RewriterBase &rewriter,
                     tensor::ExpandShapeOp expandShapeOp)
      : VisitorBase(loc, rewriter), expandShapeOp(expandShapeOp) {}
  template <typename T> LogicalResult operator()(T &self) {
    if constexpr (std::is_base_of_v<SimpleRange, T>) {
      return expandShapeRange(self);
    } else if constexpr (std::is_same_v<Mask, T>) {
      return expandShapeMask(self);
    } else if constexpr (std::is_same_v<Scalar, T>) {
      return expandShapeScalar(self);
    }
    return rewriter.notifyMatchFailure(loc, "Unsupported expand_shape");
  }

private:
  tensor::ExpandShapeOp expandShapeOp;
  LogicalResult expandShapeScalar(Scalar &self) {
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(expandShapeOp);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : axes.value()) {
      self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
    }
    return success();
  }
  LogicalResult expandShapeRange(SimpleRange &self) {
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(expandShapeOp);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : axes.value()) {
      self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
      if (self.isTrackingAxis() && axis <= self.axis)
        self.axis += 1;
    }
    return success();
  }
  LogicalResult expandShapeMask(Mask &self) {
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(expandShapeOp);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : axes.value()) {
      self.dims.insert(self.dims.begin() + axis, rewriter.getIndexAttr(1));
      self.maskStarts.insert(self.maskStarts.begin() + axis,
                             rewriter.getIndexAttr(0));
      self.maskEnds.insert(self.maskEnds.begin() + axis,
                           rewriter.getIndexAttr(1));
    }
    return success();
  }
};

struct CollapseShapeVisitor : public VisitorBase {
  CollapseShapeVisitor(Location loc, RewriterBase &rewriter,
                       tensor::CollapseShapeOp collapseShapeOp)
      : VisitorBase(loc, rewriter), collapseShapeOp(collapseShapeOp) {}
  template <typename T> LogicalResult operator()(T &self) {
    if constexpr (std::is_base_of_v<SimpleRange, T>) {
      return collapseShapeRange(self);
    } else if constexpr (std::is_same_v<Mask, T>) {
      return collapseShapeMask(self);
    } else if constexpr (std::is_same_v<Scalar, T>) {
      return collapseShapeScalar(self);
    }
    return rewriter.notifyMatchFailure(loc, "Unsupported collapse_shape");
  }

private:
  tensor::CollapseShapeOp collapseShapeOp;
  LogicalResult collapseShapeScalar(Scalar &self) {
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(collapseShapeOp);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : llvm::reverse(axes.value())) {
      self.dims.erase(self.dims.begin() + axis);
    }
    return success();
  }
  LogicalResult collapseShapeRange(SimpleRange &self) {
    // Do NOT erase the collapsed one size axis unless it is static true,
    // or it is not the simple range axis, as we do NOT known if this
    // axis is masked by false or true.
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(collapseShapeOp, self.axis);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : llvm::reverse(axes.value())) {
      assert(axis != self.axis);
      if (!isConstantIntValue(self.dims[axis], 1))
        return failure();
      self.dims.erase(self.dims.begin() + axis);
      if (self.isTrackingAxis() && axis < self.axis)
        self.axis -= 1;
    }
    return success();
  }
  LogicalResult collapseShapeMask(Mask &self) {
    std::optional<SmallVector<int64_t>> axes =
        getOneSizeAxesOfReassociativeReshapeOp(collapseShapeOp);
    if (!axes.has_value() || axes.value().empty())
      return failure();
    for (int64_t axis : llvm::reverse(axes.value())) {
      // Do NOT erase the collapsed one-sized axis unless it is static true.
      if (!isConstantIntValue(self.dims[axis], 1))
        return failure();
      self.dims.erase(self.dims.begin() + axis);
      self.maskStarts.erase(self.maskStarts.begin() + axis);
      self.maskEnds.erase(self.maskEnds.begin() + axis);
    }
    return success();
  }
};

struct TransVisitor : public VisitorBase {
  TransVisitor(Location loc, RewriterBase &rewriter, ArrayRef<int32_t> order)
      : VisitorBase(loc, rewriter), order(order) {}
  template <typename T> LogicalResult operator()(T &self) {
    return transImpl<T>(self);
  }

private:
  ArrayRef<int32_t> order;

  template <typename T> LogicalResult transImpl(T &self) {
    return rewriter.notifyMatchFailure(loc, "Unsupported trans");
  }

  SmallVector<OpFoldResult> reorder(ArrayRef<OpFoldResult> input,
                                    ArrayRef<int32_t> order) {
    SmallVector<OpFoldResult> ret(input.size());
    for (auto idx : llvm::seq<int64_t>(order.size())) {
      ret[idx] = input[order[idx]];
    }
    return ret;
  }
};

template <> LogicalResult TransVisitor::transImpl(SimpleRange &self) {
  self.dims = reorder(self.dims, order);
  if (self.isTrackingAxis()) {
    auto it = llvm::find(order, self.axis);
    assert(it != order.end() && "not found track axis");
    self.axis = std::distance(order.begin(), it);
  }
  return success();
}
template <> LogicalResult TransVisitor::transImpl(Mask &self) {
  self.dims = reorder(self.dims, order);
  self.maskStarts = reorder(self.maskStarts, order);
  self.maskEnds = reorder(self.maskEnds, order);
  return success();
}
template <> LogicalResult TransVisitor::transImpl(Scalar &self) {
  self.dims = reorder(self.dims, order);
  return success();
}

class MaskParser {
public:
  MaskParser(Location loc, RewriterBase &rewriter)
      : loc(loc), rewriter(rewriter) {}

  /// Operand is the result of a constant.
  /// Get the value of the constant and assign it to scalar.
  FailureOr<Result> parseOp(arith::ConstantOp constOp) {
    // Scalar constant will be processed in func parseIntScalar.
    auto attr = cast<DenseElementsAttr>(constOp.getValue());

    if (!attr.isSplat() || !isa<IntegerType>(attr.getElementType())) {
      return rewriter.notifyMatchFailure(
          loc, "All elements must share a single integer constant value");
    }
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto op =
        rewriter.create<arith::ConstantIndexOp>(loc, value.getSExtValue());
    Scalar ret;
    ret.scalar = op.getValue();
    if (auto tensorType = dyn_cast<TensorType>(constOp.getType())) {
      for (auto s : tensorType.getShape())
        ret.dims.push_back(rewriter.getIndexAttr(s));
    }
    return Result(ret);
  }

  FailureOr<Result> parseIntScalar(Value scalar) {
    Scalar ret;

    IntegerType scalarType = cast<IntegerType>(scalar.getType());
    if (scalarType.getWidth() == 1) {
      // Because the 1-bit integer does not have a sign bit, `IndexCastUIOp`
      // needs to be used here for conversion.
      auto castOp = rewriter.create<arith::IndexCastUIOp>(
          loc, rewriter.getIndexType(), scalar);
      ret.scalar = castOp.getResult();
    } else {
      auto castOp = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), scalar);
      ret.scalar = castOp.getResult();
    }
    return Result(ret);
  }

  /// Operand is the result of addi.
  /// One and only one of the operands should be a scalar. Increment both start
  /// and end, dims remains unchanged, and scalar is empty.
  FailureOr<Result> parseOp(arith::AddIOp addOp) {

    auto lhsState = parse(addOp.getLhs());
    if (failed(lhsState))
      return failure();

    auto rhsState = parse(addOp.getRhs());
    if (failed(rhsState))
      return failure();

    return std::visit(BinaryVisitor(loc, rewriter, addOFRs), *lhsState,
                      *rhsState);
  }

  /// Operand is the result of subi.
  /// One and only one of the operands should be a scalar. Increment both start
  /// and end, dims remains unchanged, and scalar is empty.
  FailureOr<Result> parseOp(arith::SubIOp subOp) {

    auto lhsState = parse(subOp.getLhs());
    if (failed(lhsState))
      return failure();

    auto rhsState = parse(subOp.getRhs());
    if (failed(rhsState))
      return failure();

    return std::visit(BinaryVisitor(loc, rewriter, subOFRs), *lhsState,
                      *rhsState);
  }

  /// Operand is the result of andi.
  /// Each of the result state dims is smaller of the two operands' dims.
  /// Insert instruction if needed to get new dims.
  FailureOr<Result> parseOp(arith::AndIOp andOp) {
    auto lhsState = parse(andOp.getLhs());
    if (failed(lhsState))
      return failure();

    auto rhsState = parse(andOp.getRhs());
    if (failed(rhsState))
      return failure();

    return std::visit(MergeVisitor(loc, rewriter), *lhsState, *rhsState);
  }

  /// Operand is the result of cmpi. Only support slt/sgt/sle/sgt for now.
  /// For that dimension, calculate this new dim as:
  ///   slt: dim = min(end, value) - start.
  ///   sgt: dim = end - max(start, value).
  FailureOr<Result> parseOp(arith::CmpIOp cmpOp) {
    auto cmpTy = cmpOp.getPredicate();
    if (cmpTy != arith::CmpIPredicate::slt &&
        cmpTy != arith::CmpIPredicate::sgt &&
        cmpTy != arith::CmpIPredicate::sle &&
        cmpTy != arith::CmpIPredicate::sge &&
        cmpTy != arith::CmpIPredicate::eq) {
      return rewriter.notifyMatchFailure(loc, "Unsupported cmpi predicate");
    }

    auto lhsState = parse(cmpOp.getLhs());
    if (failed(lhsState))
      return failure();

    auto rhsState = parse(cmpOp.getRhs());
    if (failed(rhsState))
      return failure();

    return std::visit(CmpVisitor(loc, rewriter, cmpTy), *lhsState, *rhsState);
  }

  /// Operand is the result of make_range.
  /// Set start and end accordingly; step size must be 1.
  FailureOr<Result> parseOp(triton::MakeRangeOp rangeOp) {
    auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
    auto start = rangeOp.getStart();
    auto end = rangeOp.getEnd();
    assert(((end - start + shape[0] - 1) / shape[0] == 1) &&
           "tt.make_range stride must be 1");

    SimpleRange ret;
    ret.start = rewriter.getIndexAttr(start);
    ret.end = rewriter.getIndexAttr(end);
    // Make range op only support 1-dim.
    ret.dims.push_back(rewriter.getIndexAttr(shape[0]));
    ret.axis = 0;

    return Result(ret);
  }

  /// Operand is the result of broadcast.
  /// Change dims only; assume only applies to tensors.
  FailureOr<Result> parseOp(triton::BroadcastOp broadcastOp) {
    auto src = broadcastOp.getSrc();
    auto dst = broadcastOp.getResult();
    // We canonicalize tt.broadcast in triton canonicalization pass,
    // so no scalar case here.
    auto dstShape = cast<ShapedType>(dst.getType()).getShape();
    auto srcShape = cast<ShapedType>(src.getType()).getShape();
    assert(srcShape.size() == dstShape.size() &&
           "rank of source and destination should match");

    auto ret = parse(src);
    if (failed(ret))
      return failure();

    if (failed(std::visit(BroadcastVisitor(loc, rewriter, srcShape, dstShape),
                          *ret)))
      return failure();

    return ret;
  }

  /// Operand is the result of splat.
  /// Assume only applies to scalar. start and end are left empty; scalar will
  /// be assigned, and dims will be updated.
  FailureOr<Result> parseOp(triton::SplatOp splatOp) {
    auto src = splatOp.getSrc();
    auto dst = splatOp.getResult();
    auto dstShape = cast<ShapedType>(dst.getType()).getShape();

    auto ret = parse(src);
    if (failed(ret))
      return failure();

    ret =
        std::visit(SplatVisitor(loc, rewriter, dstShape, src.getType()), *ret);

    return ret;
  }

  /// Operand is the result of expand_dims.
  /// Insert additional dims; start and end do not change and correspond to the
  /// dimension that contains the range.
  FailureOr<Result> parseOp(triton::ExpandDimsOp expandDimsOp) {
    auto ret = parse(expandDimsOp.getSrc());
    if (failed(ret))
      return failure();

    auto axis = expandDimsOp.getAxis();
    assert(
        cast<ShapedType>(expandDimsOp.getResult().getType()).getShape()[axis] ==
            1 &&
        "expect changed dimension to be 1 in expand_dims");

    if (failed(std::visit(ExpandDimVisitor(loc, rewriter, axis), *ret)))
      return failure();
    return ret;
  }

  /// Operand is the result of expand_shape.
  /// Insert expanded dimensions, only support expanded dimension size is one.
  /// For example:
  //// tensor<64xi32> -> tensor<1x64xi32>
  //// tensor<64xi32> -> tensor<1x64x1xi32>
  ///  tensor<1xi32> -> tensor<1x1x1xi32>
  FailureOr<Result> parseOp(tensor::ExpandShapeOp expandShapeOp) {
    auto ret = parse(expandShapeOp.getSrc());
    if (failed(ret))
      return failure();

    if (failed(
            std::visit(ExpandShapeVisitor(loc, rewriter, expandShapeOp), *ret)))
      return failure();
    return ret;
  }

  /// Operand is the result of collapse_shape.
  /// Erase collapsed dimensions, only support collapsed dimension size is one.
  /// For example:
  //// tensor<1x64xi32> -> tensor<64xi32>
  //// tensor<1x64x1xi32> -> tensor<64xi32>
  ///  tensor<1x1x1xi32> -> tensor<1xi32>
  FailureOr<Result> parseOp(tensor::CollapseShapeOp collapseShapeOp) {
    auto ret = parse(collapseShapeOp.getSrc());
    if (failed(ret))
      return failure();

    if (failed(std::visit(CollapseShapeVisitor(loc, rewriter, collapseShapeOp),
                          *ret)))
      return failure();
    return ret;
  }

  FailureOr<Result> parseOp(triton::TransOp transOp) {
    auto ret = parse(transOp.getSrc());
    if (failed(ret))
      return failure();

    if (failed(
            std::visit(TransVisitor(loc, rewriter, transOp.getOrder()), *ret)))
      return failure();

    return ret;
  }

  FailureOr<Result> parseOp(arith::ExtSIOp extsiOp) {
    return parse(extsiOp.getOperand());
  }

  FailureOr<Result> parseOp(arith::TruncIOp trunciOp) {
    return parse(trunciOp.getOperand());
  }

  FailureOr<Result> parseUnknownValue(Value operand) {
    auto type = dyn_cast<ShapedType>(operand.getType());
    if (!type)
      return rewriter.notifyMatchFailure(
          loc, "only support track shaped type value");

    assert((isa<IndexType, IntegerType, FloatType>(type.getElementType()) &&
            "unsupport unknown value type"));

    Result ret;
    // Now we think i1 is bool and is a mask, other type is unknown range.
    if (type.getElementTypeBitWidth() == 1) {
      ret = Mask(type.getRank());
    } else {
      ret = SimpleRange(type.getRank());
    }
    return ret;
  }

  FailureOr<Result> parse(Value operand) {
    if (isa<IntegerType>(operand.getType())) {
      return parseIntScalar(operand);
    }

    auto *defOp = operand.getDefiningOp();
    if (!defOp) {
      return parseUnknownValue(operand);
    }

    return llvm::TypeSwitch<Operation *, FailureOr<Result>>(defOp)
        .Case<arith::ConstantOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
              arith::SubIOp, arith::ExtSIOp, arith::TruncIOp,
              triton::MakeRangeOp, triton::BroadcastOp, triton::ExpandDimsOp,
              triton::SplatOp, triton::TransOp, tensor::CollapseShapeOp,
              tensor::ExpandShapeOp>([&](auto op) { return parseOp(op); })
        .Default([&](Operation *) { return parseUnknownValue(operand); });
  }

private:
  Location loc;
  RewriterBase &rewriter;
};
} // namespace

void MaskTracker::parse(Value operand, Location loc, RewriterBase &rewriter) {
  auto shapeTy = dyn_cast<ShapedType>(operand.getType());
  if (!shapeTy)
    return;
  int64_t rank = shapeTy.getRank();
  starts = SmallVector<OpFoldResult>(rank, nullptr);
  ends = SmallVector<OpFoldResult>(rank, nullptr);
  dims = SmallVector<OpFoldResult>(rank, nullptr);
  MaskParser parser(loc, rewriter);
  auto ret = parser.parse(operand);
  if (failed(ret) || !std::holds_alternative<Mask>(*ret))
    return;

  Mask &mask = std::get<Mask>(*ret);
  starts = mask.maskStarts;
  ends = mask.maskEnds;
  dims = mask.dims;

  LLVM_DEBUG(llvm::dbgs() << *this << "\n");
}

bool MaskTracker::hasFailedDim() const {
  return getSizes().empty() ||
         llvm::any_of(getSizes(), [](auto ofr) { return !ofr; });
}

void MaskTracker::setDimensionStatus(int64_t dim, OpFoldResult start,
                                     OpFoldResult end, OpFoldResult dimSize) {
  assert(dim >= 0 && dim < getRank());
  starts[dim] = start;
  ends[dim] = end;
  dims[dim] = dimSize;
}

LogicalResult RangeTracker::parse(Value operand, Location loc,
                                  RewriterBase &rewriter) {
  auto shapeTy = dyn_cast<ShapedType>(operand.getType());
  if (!shapeTy)
    return failure();
  MaskParser parser(loc, rewriter);
  FailureOr<Result> ret = parser.parse(operand);
  if (failed(ret) || !std::holds_alternative<SimpleRange>(*ret))
    return failure();

  SimpleRange &simpleRange = std::get<SimpleRange>(*ret);
  start = simpleRange.start;
  end = simpleRange.end;
  axis = simpleRange.axis;
  return success();
}
