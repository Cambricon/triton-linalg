//===- InferAxisInfoInterfaceImpl.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <assert.h>
#include <memory>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton-linalg/Dialect/Triton/Interfaces/InferAxisInfoInterface.h"
#include "triton-linalg/Dialect/Triton/Transforms/InferAxisInfoInterfaceImpl.h"
#include "triton-linalg/Dialect/Utils/RegistryUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

namespace mlir {
class MLIRContext;
} // namespace mlir

/// Returns the highest pow of 2 that can devide n.
/// Returns maximum number if given 0.
///
/// Example:
///  highestPowOf2Divisor<int>(0) = 1073741824
///  highestPowOf2Divisor<int>(1) = 1
///  highestPowOf2Divisor<int>(6) = 2
///  highestPowOf2Divisor<int>(8) = 8
template <typename T> static inline T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & (~(n - 1)));
}

/// If lhs * rhs overflows, return max value possible value for the type.
static int64_t multiplyDivisor(int64_t lhs, int64_t rhs) {
  int64_t maxDivisor = highestPowOf2Divisor<int64_t>(0);
  if (lhs > maxDivisor / rhs)
    return maxDivisor;
  return lhs * rhs;
}

static constexpr int log2Int(int64_t num) {
  return (num > 1) ? 1 + log2Int(num / 2) : 0;
}

static bool ltOrGtPredicate(arith::CmpIPredicate predicate) {
  return predicate == arith::CmpIPredicate::sgt ||
         predicate == arith::CmpIPredicate::ugt ||
         predicate == arith::CmpIPredicate::slt ||
         predicate == arith::CmpIPredicate::ult;
}

namespace {

//===--------------------------------------------------------------------===//
// The main logical is modified from
// lib/Analysis/AxisInfo.cpp in the triton repo.
//===--------------------------------------------------------------------===//

struct ConstantOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          ConstantOpInferAxisInfoOpInterface, arith::ConstantOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    arith::ConstantOp constantOp = cast<arith::ConstantOp>(op);
    auto intAttr = constantOp.getValue().dyn_cast<IntegerAttr>();
    if (intAttr) {
      int64_t value = intAttr.getValue().getZExtValue();

      return setResultAxisInfo(constantOp.getResult(),
                               AxisInfoExt({highestPowOf2Divisor(value)},
                                           {AxisInfoExt::kInitValue}, {0},
                                           {value}));
    }

    // TODO: generalize to dense attr.
    auto splatAttr = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.getSplatValue<APInt>().getZExtValue();
      TensorType ty = splatAttr.getType().cast<TensorType>();
      return setResultAxisInfo(
          constantOp.getResult(),
          AxisInfoExt(AxisInfoExt::DimVectorT(ty.getRank(),
                                              highestPowOf2Divisor(value)),
                      AxisInfoExt::DimVectorT(ty.getShape().begin(),
                                              ty.getShape().end()),
                      AxisInfoExt::DimVectorT(ty.getRank(), 0), {value}));
    }

    return setResultAxisInfo(constantOp.getResult(), AxisInfoExt());
  }
};

template <typename CastOpTy>
struct CastOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          CastOpInferAxisInfoOpInterface<CastOpTy>, CastOpTy> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    CastOpTy castOp = cast<CastOpTy>(op);
    assert(argInfos.size() == 1);
    return setResultAxisInfo(castOp.getResult(), argInfos[0]);
  }
};

/// Use CRTP instead of virtual function to avoid illegal vptr problems.
/// As InterfaceMap.insert() in InterfaceSupport.h directly cast concrete
/// interface class to void*, the compiler may not known
/// that the concrete interface class has vptr.
///
/// The only thing you can do with a void* pointer is to cast it back to exactly
/// the same type as the pointer that was cast to void* in the first place. The
/// behaviour on doing anything else is undefined.
template <typename OpTy, typename ConcreteType>
struct BinaryOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          BinaryOpInferAxisInfoOpInterface<OpTy, ConcreteType>, OpTy> {
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    OpTy binaryOp = cast<OpTy>(op);

    assert(argInfos.size() == 2 && "Expected two operands");

    auto &lhsInfo = argInfos[0];
    auto &rhsInfo = argInfos[1];
    auto rank = lhsInfo.getRank();
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    auto constantValue =
        static_cast<const ConcreteType *>(this)->getConstantValue(
            binaryOp, lhsInfo, rhsInfo);
    for (auto d = 0; d < rank; ++d) {
      if (constantValue.has_value()) {
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
        stride.push_back(std::max(lhsInfo.getStride(d), rhsInfo.getStride(d)));
        strideValue.push_back(0);
      } else {
        divisibility.push_back(
            static_cast<const ConcreteType *>(this)->getDivisibility(
                binaryOp, lhsInfo, rhsInfo, d));
        stride.push_back(static_cast<const ConcreteType *>(this)->getStride(
            binaryOp, lhsInfo, rhsInfo, d));
        strideValue.push_back(
            static_cast<const ConcreteType *>(this)->getStrideValue(
                binaryOp, lhsInfo, rhsInfo, d));
      }
    }
    return setResultAxisInfo(
        binaryOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

template <typename OpTy>
struct AddSubOpInferAxisInfoOpInterface
    : public BinaryOpInferAxisInfoOpInterface<
          OpTy, AddSubOpInferAxisInfoOpInterface<OpTy>> {
  int64_t getStride(OpTy op, const AxisInfoExt &lhs, const AxisInfoExt &rhs,
                    int dim) const {
    return std::gcd(lhs.getStride(dim), rhs.getStride(dim));
  }

  int64_t getStrideValue(OpTy op, const AxisInfoExt &lhs,
                         const AxisInfoExt &rhs, int dim) const {
    if (lhs.getStrideValue(dim) == AxisInfoExt::kStrideValueInitValue ||
        rhs.getStrideValue(dim) == AxisInfoExt::kStrideValueInitValue)
      return AxisInfoExt::kStrideValueInitValue;
    return std::abs(applyOp(lhs.getStrideValue(dim), rhs.getStrideValue(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfoExt &lhs,
                          const AxisInfoExt &rhs, int dim) const {
    // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
    // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
    // lhs + rhs = k * d_lhs + p * d_rhs = (k * d_lhs + p * d_rhs) * gcd(d_lhs,
    // d_rhs)
    return std::gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfoExt &lhs,
                                          const AxisInfoExt &rhs) const {
    if (!lhs.getConstantValue().has_value() ||
        !rhs.getConstantValue().has_value()) {
      return {};
    }

    return {applyOp(lhs.getConstantValue().value(),
                    rhs.getConstantValue().value())};
  }

private:
  static int64_t applyOp(int64_t lhs, int64_t rhs) {
    static_assert(std::is_same_v<OpTy, arith::SubIOp> ||
                  std::is_same_v<OpTy, arith::AddIOp> ||
                  std::is_same_v<OpTy, triton::AddPtrOp>);

    if constexpr (std::is_same_v<OpTy, arith::AddIOp> ||
                  std::is_same_v<OpTy, triton::AddPtrOp>) {
      return lhs + rhs;
    }

    return lhs - rhs;
  }
};

struct MulOpInferAxisInfoOpInterface
    : public BinaryOpInferAxisInfoOpInterface<arith::MulIOp,
                                              MulOpInferAxisInfoOpInterface> {
  int64_t getConstancy(arith::MulIOp op, const AxisInfoExt &lhs,
                       const AxisInfoExt &rhs, int dim) const {
    return std::gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  int64_t getStride(arith::MulIOp op, const AxisInfoExt &lhs,
                    const AxisInfoExt &rhs, int dim) const {
    return std::max(std::gcd(lhs.getConstancy(dim), rhs.getStride(dim)),
                    std::gcd(lhs.getStride(dim), rhs.getConstancy(dim)));
  }

  int64_t getStrideValue(arith::MulIOp op, const AxisInfoExt &lhs,
                         const AxisInfoExt &rhs, int dim) const {
    if (lhs.getStrideValue(dim) == AxisInfoExt::kStrideValueInitValue ||
        rhs.getStrideValue(dim) == AxisInfoExt::kStrideValueInitValue)
      return AxisInfoExt::kStrideValueInitValue;

    // lhs * cst
    auto lhsStrideValue =
        rhs.getConstantValue().has_value()
            ? lhs.getStrideValue(dim) * rhs.getConstantValue().value()
            : AxisInfoExt::kStrideValueInitValue;
    // cst * rhs
    auto rhsStrideValue =
        lhs.getConstantValue().has_value()
            ? rhs.getStrideValue(dim) * lhs.getConstantValue().value()
            : AxisInfoExt::kStrideValueInitValue;
    return std::max(lhsStrideValue, rhsStrideValue);
  }

  int64_t getDivisibility(arith::MulIOp op, const AxisInfoExt &lhs,
                          const AxisInfoExt &rhs, int dim) const {
    // lhs = k * d_lhs
    // rhs = p * d_rhs
    // lhs * rhs = k * d_lhs * p * d_rhs = k * p * d_lhs * d_rhs
    return multiplyDivisor(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
  }

  std::optional<int64_t> getConstantValue(arith::MulIOp op,
                                          const AxisInfoExt &lhs,
                                          const AxisInfoExt &rhs) const {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() * rhs.getConstantValue().value()};
    return {};
  }
};

template <typename OpTy>
struct DivOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          DivOpInferAxisInfoOpInterface<OpTy>, OpTy> {
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    OpTy divOp = cast<OpTy>(op);
    assert(argInfos.size() == 2 && "Expected two operands");

    auto resTy =
        divOp.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return setResultAxisInfo(divOp.getResult(), AxisInfoExt{});
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto &lhs = argInfos[0];
    auto &rhs = argInfos[1];

    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      if ((rhs.getConstantValue().has_value() &&
           rhs.getConstantValue().value() == 1) ||
          (lhs.getConstantValue().has_value() &&
           lhs.getConstantValue().value() == 0)) {
        // Case1: lhs / 1 or 0 / rhs, the result both equal to lhs.
        divisibility.push_back(lhs.getDivisibility(d));
        stride.push_back(lhs.getStride(d));
        strideValue.push_back(lhs.getStrideValue(d));
        constantValue = {lhs.getConstantValue()};
      } else if (lhs.getConstantValue().has_value() &&
                 rhs.getConstantValue().has_value()) {
        // Case2: cst1 / cst2.
        stride.push_back(lhs.getConstancy(d));
        strideValue.push_back(0);
        constantValue = {lhs.getConstantValue().value() /
                         rhs.getConstantValue().value()};
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
      } else if (!lhs.isFullConstantDim(shape, d) &&
                 lhs.isFullStrideDim(shape, d) &&
                 rhs.isFullConstantDim(shape, d) &&
                 rhs.getConstantValue().has_value() &&
                 llvm::isPowerOf2_64(lhs.getStrideValue(d))) {
        // Case 3: lhs stride(stride_val is power of 2), rhs constant.
        // lhs: d_lhs * k, d_lhs * k + s, ..., d_lhs * k + n * s
        // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
        // lhs / rhs = d_lhs * k / (d_rhs * p), (d_lhs * k + s) / (d_rhs * p),
        // ..., (d_lhs * k + n*s) / (d_rhs * p)
        // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
        // the minimal stride is
        // minStride = max(gcd(d_lhs, d_rhs) / strideVal, 1).
        // Since minStride maybe > len(lhs),
        // we need to use another gcd to get the actual constancy.
        int64_t divisibilityGCD =
            std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
        bool isFullStrided =
            lhs.getStrideValue(d) % rhs.getConstantValue().value() == 0;
        int64_t newStride =
            isFullStrided
                ? lhs.getStride(d)
                : std::max<int64_t>(divisibilityGCD / lhs.getStrideValue(d), 1);
        stride.push_back(std::gcd(lhs.getStride(d), newStride));
        strideValue.push_back(lhs.getStrideValue(d) /
                              rhs.getConstantValue().value());
        divisibility.push_back(std::max<int64_t>(
            lhs.getDivisibility(d) / rhs.getConstantValue().value(), 1));
      } else if (lhs.isStridedConstantDim(shape, d) &&
                 rhs.getConstantValue().has_value()) {
        divisibility.push_back(std::max<int64_t>(
            lhs.getDivisibility(d) / rhs.getConstantValue().value(), 1));
        stride.push_back(std::gcd(lhs.getStride(d), rhs.getStride(d)));
        strideValue.push_back(0);
      } else {
        divisibility.push_back(AxisInfoExt::kInitValue);
        stride.push_back(AxisInfoExt::kInitValue);
        strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
      }
    }

    return setResultAxisInfo(
        divOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

template <typename OpTy>
struct RemOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          RemOpInferAxisInfoOpInterface<OpTy>, OpTy> {
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    OpTy remOp = cast<OpTy>(op);
    assert(argInfos.size() == 2 && "Expected two operands");

    auto resTy =
        remOp.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return setResultAxisInfo(remOp.getResult(), AxisInfoExt{});
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto &lhs = argInfos[0];
    auto &rhs = argInfos[1];

    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      if (rhs.getConstantValue().has_value() &&
          rhs.getConstantValue().value() == 1) {
        // Case1: lhs % 1.
        divisibility.push_back(highestPowOf2Divisor<int64_t>(0));
        stride.push_back(shape[d]);
        strideValue.push_back(0);
        constantValue = {0};
      } else if (lhs.getConstantValue().has_value() &&
                 rhs.getConstantValue().has_value()) {
        // Case2: cst1 % cst2.
        stride.push_back(lhs.getConstancy(d));
        strideValue.push_back(0);
        constantValue = {lhs.getConstantValue().value() %
                         rhs.getConstantValue().value()};
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
      } else if (lhs.isFullContiguousDim(shape, d) &&
                 rhs.isFullConstantDim(shape, d)) {
        // Case3: lhs contiguous, rhs constant.
        // lhs: d_lhs * k = gcd(d_lhs, d_rhs) * k' * k = gcd(d_lhs, d_rhs) * k''
        // rhs: d_rhs * p = gcd(d_lhs, d_rhs) * p' * p = gcd(d_lhs, d_rhs) * p''
        // lhs = gcd(d_lhs, d_rhs) * k'' = gcd(d_lhs, d_rhs) * d + r
        // r must be divisible by gcd(d_lhs, d_rhs)
        divisibility.push_back(
            std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));

        // lhs: d_lhs * k, d_lhs * k + 1, ..., d_lhs * k + n
        // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
        // lhs % rhs = d_lhs * k % (d_rhs * p), (d_lhs * k + 1) % (d_rhs * p),
        // ..., (d_lhs * k + n) % (d_rhs * p)
        // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
        // The minimal contiguity is gcd(d_lhs, d_rhs).
        // Since gcd(d_lhs, d_rhs) maybe > len(lhs),
        // we need to use another gcd to get the actual contiguity.
        stride.push_back(
            std::gcd(lhs.getContiguity(d),
                     std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d))));
        strideValue.push_back(1);
      } else if (lhs.isStridedContiguousDim(shape, d) &&
                 rhs.getConstantValue().has_value()) {
        // Case4: lhs strided contiguous, rhs constant value.
        divisibility.push_back(
            std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
        stride.push_back(
            std::gcd(lhs.getContiguity(d),
                     std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d))));
        strideValue.push_back(lhs.getStrideValue(d) %
                              rhs.getConstantValue().value());
      } else if (lhs.isStridedConstantDim(shape, d) &&
                 rhs.getConstantValue().has_value()) {
        // Case5: lhs strided constant, rhs constant value.
        divisibility.push_back(
            std::gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
        stride.push_back(lhs.getConstancy(d));
        strideValue.push_back(0);
      } else {
        divisibility.push_back(AxisInfoExt::kInitValue);
        stride.push_back(AxisInfoExt::kInitValue);
        strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
      }
    }

    return setResultAxisInfo(
        remOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

struct CmpOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          CmpOpInferAxisInfoOpInterface, arith::CmpIOp> {
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    arith::CmpIOp cmpOp = cast<arith::CmpIOp>(op);
    assert(argInfos.size() == 2 && "Expected two operands");

    auto resTy = cmpOp.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resTy)
      return setResultAxisInfo(cmpOp.getResult(), AxisInfoExt{});
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto &lhsInfo = argInfos[0];
    auto &rhsInfo = argInfos[1];

    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      int64_t constancyHint = AxisInfoExt::kInitValue;
      int64_t strideValueHint = AxisInfoExt::kStrideValueInitValue;
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        constancyHint = lhsInfo.getConstancy(d);
        constantValue = arith::applyCmpPredicate(
                            cmpOp.getPredicate(),
                            APInt(64, lhsInfo.getConstantValue().value()),
                            APInt(64, rhsInfo.getConstantValue().value()))
                            ? 1
                            : 0;
        strideValueHint = 0;
      } else if (ltOrGtPredicate(cmpOp.getPredicate())) {
        // Lhs and rhs are both partial constants.
        constancyHint =
            std::gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));

        auto commonDivisor =
            std::gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d));
        if (lhsInfo.isFullConstantDim(shape, d) &&
            rhsInfo.isFullContiguousDim(shape, d)) {
          // Case 2: lhs all constant, rhs all contiguous
          // NOTE:
          // lhs: k0 * d, k0 * d, ...
          // rhs: k1 * d, k1 * d + 1, ...
          // lhs lt rhs: 1, 1, 1, 1 (minimal len: d if k0 <= k1)
          // lhs lt rhs: 0, 0, 0, 0 (minimal len: d if k0 > k1)
          // lhs gt rhs: 0, 0, 0, 0 (minimal len: d if k0 <= k1)
          // lhs gt rhs: 1, 1, 1, 1 (minimal len: d if k0 > k1)
          constancyHint = std::max(
              constancyHint, std::gcd(rhsInfo.getContiguity(d), commonDivisor));
        } else if (lhsInfo.isFullContiguousDim(shape, d) &&
                   rhsInfo.isFullConstantDim(shape, d)) {
          // Case 3: lhs all contiguous, rhs all constant
          // NOTE
          // lhs: k0 * d, k0 * d + 1, ...
          // rhs: k1 * d, k1 * d, ...
          // lhs gt rhs: 1, 1, 1, 1 (minimal len: d if k0 >= k1)
          // lhs gt rhs: 0, 0, 0, 0 (minimal len: d if k0 < k1)
          // lhs lt rhs: 0, 0, 0, 0 (minimal len: d if k0 >= k1)
          // lhs lt rhs: 1, 1, 1, 1 (minimal len: d if k0 < k1)
          constancyHint = std::max(
              constancyHint, std::gcd(lhsInfo.getContiguity(d), commonDivisor));
        } else if (lhsInfo.isFullConstantDim(shape, d) &&
                   rhsInfo.isFullConstantDim(shape, d)) {
          // Case 4: lhs all constant, rhs all constant
          strideValueHint = 0;
        }
      }

      divisibility.push_back(AxisInfoExt::kInitValue);
      stride.push_back(constancyHint);
      strideValue.push_back(strideValueHint);
    }

    return setResultAxisInfo(
        cmpOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

struct ShLIOpInferAxisInfoOpInterface
    : public BinaryOpInferAxisInfoOpInterface<arith::ShLIOp,
                                              ShLIOpInferAxisInfoOpInterface> {
  int64_t getStride(arith::ShLIOp op, const AxisInfoExt &lhs,
                    const AxisInfoExt &rhs, int dim) const {
    if (rhs.getConstantValue().has_value())
      return lhs.getContiguity(dim);
    return AxisInfoExt::kInitValue;
  }

  int64_t getStrideValue(arith::ShLIOp op, const AxisInfoExt &lhs,
                         const AxisInfoExt &rhs, int dim) const {
    if (rhs.getConstantValue().has_value()) {
      auto shift = rhs.getConstantValue().value();
      auto numBits = log2Int(shift);
      auto maxBits = log2Int(highestPowOf2Divisor<int64_t>(0));
      if (shift + numBits > maxBits)
        return AxisInfoExt::kStrideValueInitValue;
      return lhs.getStrideValue(dim) << rhs.getConstantValue().value();
    }
    return AxisInfoExt::kStrideValueInitValue;
  }

  int64_t getDivisibility(arith::ShLIOp op, const AxisInfoExt &lhs,
                          const AxisInfoExt &rhs, int dim) const {
    auto shift = rhs.getConstantValue().has_value()
                     ? rhs.getConstantValue().value()
                     : rhs.getDivisibility(dim);
    auto numBits = log2Int(lhs.getDivisibility(dim));
    auto maxBits = log2Int(highestPowOf2Divisor<int64_t>(0));
    // Make sure the return value doesn't exceed
    // highestPowOf2Divisor<int64>(0).
    if (shift + numBits > maxBits)
      return highestPowOf2Divisor<int64_t>(0);
    return lhs.getDivisibility(dim) << shift;
  }

  std::optional<int64_t> getConstantValue(arith::ShLIOp op,
                                          const AxisInfoExt &lhs,
                                          const AxisInfoExt &rhs) const {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() << rhs.getConstantValue().value()};
    return {};
  }
};

template <typename OpTy>
struct ShROpInferAxisInfoOpInterface
    : public BinaryOpInferAxisInfoOpInterface<
          OpTy, ShROpInferAxisInfoOpInterface<OpTy>> {
  int64_t getStride(OpTy op, const AxisInfoExt &lhs, const AxisInfoExt &rhs,
                    int dim) const {
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      return lhs.getContiguity(dim);
    return AxisInfoExt::kInitValue;
  }

  int64_t getStrideValue(OpTy op, const AxisInfoExt &lhs,
                         const AxisInfoExt &rhs, int dim) const {
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      return lhs.getStrideValue(dim);
    return AxisInfoExt::kStrideValueInitValue;
  }

  int64_t getDivisibility(OpTy op, const AxisInfoExt &lhs,
                          const AxisInfoExt &rhs, int dim) const {
    if (rhs.getConstantValue().has_value())
      return std::max<int64_t>(AxisInfoExt::kInitValue,
                               lhs.getDivisibility(dim) /
                                   (1 << rhs.getConstantValue().value()));
    return AxisInfoExt::kInitValue;
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfoExt &lhs,
                                          const AxisInfoExt &rhs) const {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() >> rhs.getConstantValue().value()};
    return {};
  }
};

struct SelectOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          SelectOpInferAxisInfoOpInterface, arith::SelectOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    arith::SelectOp selectOp = cast<arith::SelectOp>(op);
    assert(argInfos.size() == 3 && "Expected three operands");

    auto resTy =
        selectOp.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return setResultAxisInfo(selectOp.getResult(), AxisInfoExt());
    auto shape = resTy.getShape();
    auto rank = shape.size();
    auto &lhsInfo = argInfos[1];
    auto &rhsInfo = argInfos[2];

    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    std::optional<int64_t> constantValue;
    if (argInfos[0].getConstantValue().has_value()) {
      if (argInfos[0].getConstantValue() == 0) {
        divisibility = rhsInfo.getDivisibility();
        stride = rhsInfo.getStride();
        strideValue = rhsInfo.getStrideValue();
        constantValue = rhsInfo.getConstantValue();
      } else {
        divisibility = lhsInfo.getDivisibility();
        stride = lhsInfo.getStride();
        strideValue = lhsInfo.getStrideValue();
        constantValue = lhsInfo.getConstantValue();
      }
    } else {
      bool i1Cond = isa<IntegerType>(op->getOperand(0).getType());
      for (auto d = 0; d < rank; ++d) {
        if (i1Cond) {
          stride.push_back(
              std::gcd(lhsInfo.getStride(d), rhsInfo.getStride(d)));
        } else {
          stride.push_back(std::gcd(
              std::gcd(lhsInfo.getStride(d), argInfos[0].getConstancy(d)),
              std::gcd(rhsInfo.getStride(d), argInfos[0].getConstancy(d))));
        }
        strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
        divisibility.push_back(
            std::min(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
      }
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value() &&
          lhsInfo.getConstantValue() == rhsInfo.getConstantValue())
        constantValue = lhsInfo.getConstantValue();
    }

    return setResultAxisInfo(
        selectOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

template <typename OpTy>
struct MaxMinLogicalOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          MaxMinLogicalOpInferAxisInfoOpInterface<OpTy>, OpTy> {
  static_assert(llvm::is_one_of<OpTy, arith::MaxSIOp, arith::MaxUIOp,
                                arith::MinSIOp, arith::MinUIOp, arith::AndIOp,
                                arith::OrIOp, arith::XOrIOp>::value);
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    OpTy concretOp = cast<OpTy>(op);
    assert(argInfos.size() == 2 && "Expected two operands");
    auto &lhsInfo = argInfos[0];
    auto &rhsInfo = argInfos[1];

    std::optional<int64_t> constantValue = std::nullopt;
    AxisInfoExt::DimVectorT stride, strideValue, divisibility;
    auto rank = lhsInfo.getRank();
    for (int d = 0; d < rank; ++d) {
      const AxisInfoExt *resInfo = nullptr;
      if constexpr (llvm::is_one_of<OpTy, arith::MaxSIOp,
                                    arith::MaxUIOp>::value) {
        if (lhsInfo.getConstantValue().has_value() &&
            rhsInfo.getConstantValue().has_value()) {
          constantValue = {std::max(lhsInfo.getConstantValue().value(),
                                    rhsInfo.getConstantValue().value())};
          if (lhsInfo.getConstantValue().value() >=
              rhsInfo.getConstantValue().value()) {
            resInfo = &lhsInfo;
          } else {
            resInfo = &rhsInfo;
          }
          divisibility.push_back(resInfo->getDivisibility(d));
          stride.push_back(resInfo->getStride(d));
          strideValue.push_back(resInfo->getStrideValue(d));
          continue;
        }
      }
      if constexpr (llvm::is_one_of<OpTy, arith::MinSIOp,
                                    arith::MinUIOp>::value) {
        if (lhsInfo.getConstantValue().has_value() &&
            rhsInfo.getConstantValue().has_value()) {
          constantValue = {std::min(lhsInfo.getConstantValue().value(),
                                    rhsInfo.getConstantValue().value())};
          if (lhsInfo.getConstantValue().value() <=
              rhsInfo.getConstantValue().value()) {
            resInfo = &lhsInfo;
          } else {
            resInfo = &rhsInfo;
          }
          divisibility.push_back(resInfo->getDivisibility(d));
          stride.push_back(resInfo->getStride(d));
          strideValue.push_back(resInfo->getStrideValue(d));
          continue;
        }
      }

      if constexpr (llvm::is_one_of<OpTy, arith::AndIOp, arith::OrIOp,
                                    arith::XOrIOp>::value) {
        if (lhsInfo.getConstantValue().has_value() &&
            rhsInfo.getConstantValue().has_value()) {
          if constexpr (std::is_same_v<OpTy, arith::AndIOp>) {
            constantValue = {lhsInfo.getConstantValue().value() &
                             rhsInfo.getConstantValue().value()};
          } else if constexpr (std::is_same_v<OpTy, arith::OrIOp>) {
            constantValue = {lhsInfo.getConstantValue().value() |
                             rhsInfo.getConstantValue().value()};
          } else if constexpr (std::is_same_v<OpTy, arith::XOrIOp>) {
            constantValue = {lhsInfo.getConstantValue().value() ^
                             rhsInfo.getConstantValue().value()};
          }
        }
        if (lhsInfo.getStrideValue(d) == 0 && rhsInfo.getStrideValue(d) == 0) {
          divisibility.push_back(
              std::gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
          stride.push_back(
              std::gcd(lhsInfo.getStride(d), rhsInfo.getStride(d)));
          strideValue.push_back(0);
          continue;
        }
      }

      divisibility.push_back(AxisInfoExt::kInitValue);
      stride.push_back(AxisInfoExt::kInitValue);
      strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
    }
    return setResultAxisInfo(
        concretOp.getResult(),
        AxisInfoExt(divisibility, stride, strideValue, constantValue));
  }
};

struct MakeRangeOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          MakeRangeOpInferAxisInfoOpInterface, triton::MakeRangeOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    triton::MakeRangeOp makeRangeOp = cast<triton::MakeRangeOp>(op);
    auto start = makeRangeOp.getStart();
    auto end = makeRangeOp.getEnd();
    return setResultAxisInfo(
        makeRangeOp.getResult(),
        AxisInfoExt({highestPowOf2Divisor(start)}, {end - start}, {1}));
  }
};

struct BroadcastOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          BroadcastOpInferAxisInfoOpInterface, triton::BroadcastOp> {
  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    triton::BroadcastOp broadcastOp = cast<triton::BroadcastOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");
    TensorType retTy = broadcastOp.getResult().getType().cast<TensorType>();
    ArrayRef<int64_t> retShape = retTy.getShape();
    TensorType opTy =
        broadcastOp.getOperand().getType().dyn_cast_or_null<TensorType>();
    ArrayRef<int64_t> opShape;
    SmallVector<int64_t> scalarAsShapeOne(retTy.getRank(), 1);
    if (opTy) {
      opShape = opTy.getShape();
    } else {
      opShape = scalarAsShapeOne;
    }
    assert(opShape.size() == retTy.getRank() &&
           "Input rank should be equal with output rank if input is not scalar "
           "for tt.broadcast");
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    for (int d = 0; d < retTy.getRank(); ++d) {
      if (opTy) {
        divisibility.push_back(argInfos[0].getDivisibility(d));
      } else {
        divisibility.push_back(argInfos[0].getDivisibility(0));
      }
      stride.push_back(opShape[d] == 1 ? retShape[d]
                                       : argInfos[0].getStride(d));
      strideValue.push_back(opShape[d] == 1 ? 0
                                            : argInfos[0].getStrideValue(d));
    }

    return setResultAxisInfo(broadcastOp.getResult(),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         argInfos[0].getConstantValue()));
  }
};

struct SplatOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          SplatOpInferAxisInfoOpInterface, triton::SplatOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    triton::SplatOp splatOp = cast<triton::SplatOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");
    TensorType retTy = splatOp.getResult().getType().cast<TensorType>();
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    for (int d = 0; d < retTy.getRank(); ++d) {
      divisibility.push_back(argInfos[0].getDivisibility(0));
      stride.push_back(retTy.getShape()[d]);
      strideValue.push_back(0);
    }
    return setResultAxisInfo(splatOp.getResult(),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         argInfos[0].getConstantValue()));
  }
};

struct ExpandDimsOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          ExpandDimsOpInferAxisInfoOpInterface, triton::ExpandDimsOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    triton::ExpandDimsOp expandDimsOp = cast<triton::ExpandDimsOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");

    auto &opInfo = argInfos[0];
    AxisInfoExt::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfoExt::DimVectorT stride = opInfo.getStride();
    AxisInfoExt::DimVectorT strideValue = opInfo.getStrideValue();
    ArrayRef<int64_t> srcShape = expandDimsOp.getSrc().getType().getShape();
    int64_t expandedDim =
        std::max(static_cast<int32_t>(expandDimsOp.getAxis()) - 1, 0);
    int64_t expandedDivisibility =
        opInfo.isFullConstantDim(srcShape, expandedDim)
            ? divisibility[expandedDim]
            : AxisInfoExt::kInitValue;
    divisibility.insert(divisibility.begin() + expandDimsOp.getAxis(),
                        expandedDivisibility);
    stride.insert(stride.begin() + expandDimsOp.getAxis(),
                  AxisInfoExt::kInitValue);
    strideValue.insert(strideValue.begin() + expandDimsOp.getAxis(),
                       AxisInfoExt::kStrideValueInitValue);
    return setResultAxisInfo(expandDimsOp->getResult(0),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         opInfo.getConstantValue()));
  }
};

struct ExpandShapeOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          ExpandShapeOpInferAxisInfoOpInterface, tensor::ExpandShapeOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    tensor::ExpandShapeOp expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");

    AxisInfoExt opInfo = argInfos[0];
    ArrayRef<int64_t> srcShape = expandShapeOp.getSrcType().getShape();
    ArrayRef<int64_t> resShape = expandShapeOp.getResultType().getShape();
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    for (const auto &indices :
         llvm::enumerate(expandShapeOp.getReassociationIndices())) {
      // Init expanded axisinfo by source axisinfo.
      int64_t srcDim = indices.index();
      int64_t srcDivisibility = opInfo.getDivisibility()[srcDim];
      int64_t srcStride = opInfo.getStride()[srcDim];
      int64_t srcStrideValue = opInfo.getStrideValue()[srcDim];
      for (auto indice : indices.value()) {
        stride.push_back(srcStride);
        strideValue.push_back(srcStrideValue);
        divisibility.push_back(srcDivisibility);
      }
      if (indices.value().size() == 1)
        continue;

      // Calculate axisinfo of expanded dimension.
      int64_t nextStride = srcStride;
      int64_t nextStrideValue = srcStrideValue;
      int64_t nextDivisibility = srcDivisibility;
      for (auto resDim : llvm::reverse(indices.value())) {
        if (nextStride >= resShape[resDim] &&
            nextStride % resShape[resDim] == 0) {
          strideValue[resDim] = nextStrideValue;
          nextStrideValue = nextStride == resShape[resDim]
                                ? AxisInfoExt::kStrideValueInitValue
                                : nextStrideValue * resShape[resDim];
          stride[resDim] = resShape[resDim];
          nextStride /= resShape[resDim];
          if (opInfo.isFullConstantDim(srcShape, srcDim)) {
            divisibility[resDim] = nextDivisibility;
          } else if (opInfo.isNonConstantFullStrideDim(srcShape, srcDim)) {
            divisibility[resDim] = std::gcd(
                nextDivisibility, resShape[resDim] * strideValue[resDim]);
            nextDivisibility = srcStrideValue == 1
                                   ? AxisInfoExt::kInitValue
                                   : highestPowOf2Divisor(strideValue[resDim]);
          } else {
            divisibility[resDim] = AxisInfoExt::kInitValue;
            nextDivisibility = AxisInfoExt::kInitValue;
          }
        } else if (resShape[resDim] > nextStride &&
                   resShape[resDim] % nextStride == 0) {
          strideValue[resDim] = nextStrideValue;
          nextStrideValue = AxisInfoExt::kStrideValueInitValue;
          stride[resDim] = nextStride;
          nextStride = AxisInfoExt::kInitValue;
          divisibility[resDim] = AxisInfoExt::kInitValue;
          nextDivisibility = AxisInfoExt::kInitValue;
        } else {
          strideValue[resDim] = AxisInfoExt::kStrideValueInitValue;
          nextStrideValue = AxisInfoExt::kStrideValueInitValue;
          stride[resDim] = AxisInfoExt::kInitValue;
          nextStride = AxisInfoExt::kInitValue;
          divisibility[resDim] = AxisInfoExt::kInitValue;
          nextDivisibility = AxisInfoExt::kInitValue;
        }
      }
    }

    return setResultAxisInfo(expandShapeOp->getResult(0),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         opInfo.getConstantValue()));
  }
};

struct CollapseShapeOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          CollapseShapeOpInferAxisInfoOpInterface, tensor::CollapseShapeOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    tensor::CollapseShapeOp collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");

    AxisInfoExt opInfo = argInfos[0];
    ArrayRef<int64_t> srcShape = collapseShapeOp.getSrcType().getShape();
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    for (const auto &indices :
         llvm::enumerate(collapseShapeOp.getReassociationIndices())) {
      int64_t resDim = indices.value().back();
      int64_t resDivisibility = opInfo.getDivisibility()[resDim];
      int64_t resStride = opInfo.getStride()[resDim];
      int64_t resStrideValue = opInfo.getStrideValue()[resDim];
      for (const auto &indice :
           llvm::enumerate(llvm::reverse(indices.value()))) {
        if (indices.value().size() == 1 || indice.index() == 0)
          continue;
        int64_t srcDim = indice.value();
        int64_t srcStride = opInfo.getStride()[srcDim];
        int64_t srcStrideValue = opInfo.getStrideValue()[srcDim];
        int64_t srcDivisibility = opInfo.getDivisibility()[srcDim];
        bool isLastDimFullStrided =
            opInfo.getStride()[srcDim + 1] == srcShape[srcDim + 1];
        if (srcStrideValue == resStride * resStrideValue &&
            isLastDimFullStrided) {
          resStride *= srcStride;
        }
        resDivisibility = std::max(resDivisibility, srcDivisibility);
      }
      divisibility.push_back(resDivisibility);
      stride.push_back(resStride);
      strideValue.push_back(resStrideValue);
    }

    return setResultAxisInfo(collapseShapeOp->getResult(0),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         opInfo.getConstantValue()));
  }
};

struct ExtractSliceOpInferAxisInfoOpInterface
    : public InferAxisInfoInterface::ExternalModel<
          ExtractSliceOpInferAxisInfoOpInterface, tensor::ExtractSliceOp> {

  void inferAxisInfos(Operation *op, ArrayRef<AxisInfoExt> argInfos,
                      SetAxisInfoFn setResultAxisInfo) const {
    tensor::ExtractSliceOp extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    assert(argInfos.size() == 1 && "Expected one operand");

    AxisInfoExt opInfo = argInfos[0];
    ArrayRef<int64_t> extractSliceOffsets = extractSliceOp.getStaticOffsets();
    ArrayRef<int64_t> extractSliceSizes = extractSliceOp.getStaticSizes();
    ArrayRef<int64_t> extractSliceStrides = extractSliceOp.getStaticStrides();
    ArrayRef<int64_t> srcShape = extractSliceOp.getSourceType().getShape();
    AxisInfoExt::DimVectorT divisibility, stride, strideValue;
    int64_t extractSliceRank = extractSliceOp.getResultType().getRank();
    for (int64_t d = 0; d < extractSliceRank; ++d) {
      bool isSliceInsideFirstStride =
          (extractSliceOffsets[d] * extractSliceStrides[d] +
           (extractSliceSizes[d] - 1) * extractSliceStrides[d]) <=
          opInfo.getStride()[d];
      bool isSliceInsideOtherStride =
          extractSliceOffsets[d] * extractSliceStrides[d] >=
              opInfo.getStride()[d] &&
          (extractSliceOffsets[d] * extractSliceStrides[d] %
               opInfo.getStride()[d] +
           (extractSliceSizes[d] - 1) * extractSliceStrides[d]) <=
              opInfo.getStride()[d];
      bool isSliceMultipleStride =
          extractSliceStrides[d] == 1 &&
          extractSliceOffsets[d] % opInfo.getStride()[d] == 0 &&
          extractSliceSizes[d] % opInfo.getStride()[d] == 0 &&
          extractSliceSizes[d] / opInfo.getStride()[d] >= 1;
      if (opInfo.isStridedConstantDim(srcShape, d)) {
        if (opInfo.isFullStrideDim(srcShape, d) || isSliceInsideFirstStride) {
          stride.push_back(extractSliceSizes[d]);
          strideValue.push_back(opInfo.getStrideValue()[d]);
          divisibility.push_back(opInfo.getDivisibility()[d]);
        } else if (isSliceInsideOtherStride) {
          stride.push_back(extractSliceSizes[d]);
          strideValue.push_back(opInfo.getStrideValue()[d]);
          divisibility.push_back(AxisInfoExt::kInitValue);
        } else if (isSliceMultipleStride) {
          stride.push_back(opInfo.getStride()[d]);
          strideValue.push_back(opInfo.getStrideValue()[d]);
          divisibility.push_back(extractSliceOffsets[d] == 0
                                     ? opInfo.getDivisibility()[d]
                                     : AxisInfoExt::kInitValue);
        } else {
          stride.push_back(AxisInfoExt::kInitValue);
          strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
          divisibility.push_back(extractSliceOffsets[d] == 0
                                     ? opInfo.getDivisibility()[d]
                                     : AxisInfoExt::kInitValue);
        }
      } else if (opInfo.isNonStridedConstantStrideDim(srcShape, d)) {
        if (opInfo.isFullStrideDim(srcShape, d) || isSliceInsideFirstStride) {
          stride.push_back(extractSliceSizes[d]);
          strideValue.push_back(opInfo.getStrideValue()[d] *
                                extractSliceStrides[d]);
          divisibility.push_back(extractSliceOffsets[d] == 0
                                     ? opInfo.getDivisibility()[d]
                                     : AxisInfoExt::kInitValue);
        } else if (isSliceInsideOtherStride) {
          stride.push_back(extractSliceSizes[d]);
          strideValue.push_back(opInfo.getStrideValue()[d] *
                                extractSliceStrides[d]);
          divisibility.push_back(AxisInfoExt::kInitValue);
        } else if (isSliceMultipleStride) {
          stride.push_back(opInfo.getStride()[d]);
          strideValue.push_back(opInfo.getStrideValue()[d]);
          divisibility.push_back(extractSliceOffsets[d] == 0
                                     ? opInfo.getDivisibility()[d]
                                     : AxisInfoExt::kInitValue);
        } else {
          stride.push_back(AxisInfoExt::kInitValue);
          strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
          divisibility.push_back(AxisInfoExt::kInitValue);
        }
      } else {
        stride.push_back(AxisInfoExt::kInitValue);
        strideValue.push_back(AxisInfoExt::kStrideValueInitValue);
        divisibility.push_back(AxisInfoExt::kInitValue);
      }
    }
    return setResultAxisInfo(extractSliceOp.getResult(),
                             AxisInfoExt(divisibility, stride, strideValue,
                                         opInfo.getConstantValue()));
  }
};

} // anonymous namespace

void mlir::triton::registerInferAxisInfoInterfaceExternalModels(
    DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<triton::TritonDialect, arith::ArithDialect,
                  tensor::TensorDialect>();

  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::ConstantOp::attachInterface<ConstantOpInferAxisInfoOpInterface>(
        *ctx);
    arith::MulIOp::attachInterface<MulOpInferAxisInfoOpInterface>(*ctx);
    arith::SelectOp::attachInterface<SelectOpInferAxisInfoOpInterface>(*ctx);
    arith::CmpIOp::attachInterface<CmpOpInferAxisInfoOpInterface>(*ctx);
    arith::ShLIOp::attachInterface<ShLIOpInferAxisInfoOpInterface>(*ctx);

    utils::AttachInterfaceHelper<CastOpInferAxisInfoOpInterface, arith::ExtSIOp,
                                 arith::ExtUIOp, arith::TruncIOp,
                                 arith::IndexCastOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<RemOpInferAxisInfoOpInterface, arith::RemSIOp,
                                 arith::RemUIOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<AddSubOpInferAxisInfoOpInterface,
                                 arith::AddIOp,
                                 arith::SubIOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<MaxMinLogicalOpInferAxisInfoOpInterface,
                                 arith::AndIOp, arith::OrIOp,
                                 arith::XOrIOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<DivOpInferAxisInfoOpInterface, arith::DivSIOp,
                                 arith::DivUIOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<ShROpInferAxisInfoOpInterface, arith::ShRSIOp,
                                 arith::ShRUIOp>::registerOpInterface(ctx);
    utils::AttachInterfaceHelper<MaxMinLogicalOpInferAxisInfoOpInterface,
                                 arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
                                 arith::MinUIOp>::registerOpInterface(ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, triton::TritonDialect *dialect) {
    utils::AttachInterfaceHelper<CastOpInferAxisInfoOpInterface,
                                 triton::IntToPtrOp, triton::PtrToIntOp,
                                 triton::BitcastOp>::registerOpInterface(ctx);

    triton::MakeRangeOp::attachInterface<MakeRangeOpInferAxisInfoOpInterface>(
        *ctx);
    triton::AddPtrOp::attachInterface<
        AddSubOpInferAxisInfoOpInterface<triton::AddPtrOp>>(*ctx);

    triton::BroadcastOp::attachInterface<BroadcastOpInferAxisInfoOpInterface>(
        *ctx);
    triton::SplatOp::attachInterface<SplatOpInferAxisInfoOpInterface>(*ctx);
    triton::ExpandDimsOp::attachInterface<ExpandDimsOpInferAxisInfoOpInterface>(
        *ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::ExtractSliceOp::attachInterface<
        ExtractSliceOpInferAxisInfoOpInterface>(*ctx);
    tensor::ExpandShapeOp::attachInterface<
        ExpandShapeOpInferAxisInfoOpInterface>(*ctx);
    tensor::CollapseShapeOp::attachInterface<
        CollapseShapeOpInferAxisInfoOpInterface>(*ctx);
  });
}
