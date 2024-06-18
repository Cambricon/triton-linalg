//===- AtomicRmwConversion.cpp - atomicRmw op conversion---------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <optional>
#include <stdint.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/AtomicRmwConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TritonPointerConversion.h"
#include "triton-linalg/Conversion/TritonToLinalg/TypeConverter.h"
#include "triton-linalg/Conversion/TritonToLinalg/Utils.h"
#include "triton-linalg/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "triton-linalg/Dialect/Triton/Utils/PointerMetaInfoTracker.h"
#include "triton-linalg/Dialect/Utils/Conventions.h"
#include "triton-linalg/Dialect/Utils/ShapeUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
class MLIRContext;
class DataFlowSolver;
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
/// Try to convert triton::RMWOp to triton::linalg_ext::AtomicType.
static std::optional<triton::linalg_ext::AtomicType>
getAtomicRMWType(triton::RMWOp rmwOp, Type eltTy) {
  switch (rmwOp) {
  case triton::RMWOp::AND:
    return triton::linalg_ext::AtomicType::andi;
  case triton::RMWOp::OR:
    return triton::linalg_ext::AtomicType::ori;
  case triton::RMWOp::XOR:
    return triton::linalg_ext::AtomicType::xori;
  case triton::RMWOp::ADD:
    return triton::linalg_ext::AtomicType::addi;
  case triton::RMWOp::FADD:
    return triton::linalg_ext::AtomicType::addf;
  case triton::RMWOp::MAX:
    if (eltTy.isIntOrIndex())
      return triton::linalg_ext::AtomicType::maxs;
    return triton::linalg_ext::AtomicType::maximumf;
  case triton::RMWOp::MIN:
    if (eltTy.isIntOrIndex())
      return triton::linalg_ext::AtomicType::mins;
    return triton::linalg_ext::AtomicType::minimumf;
  case triton::RMWOp::UMAX:
    return triton::linalg_ext::AtomicType::maxu;
  case triton::RMWOp::UMIN:
    return triton::linalg_ext::AtomicType::minu;
  case triton::RMWOp::XCHG:
    return triton::linalg_ext::AtomicType::xchg;
  default:
    llvm_unreachable("Invalid AtomicRMWKind");
  }
}

/// Scalar atomic.
class TritonScalarAtomicRMWOpConversion
    : public OpConversionPattern<triton::AtomicRMWOp>,
      public triton::TritonPtrScalarConversionBase {
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

public:
  TritonScalarAtomicRMWOpConversion(
      mlir::triton::TritonLinalgTypeConverter &converter,
      mlir::MLIRContext *context, mlir::DataFlowSolver &solver,
      mlir::PatternBenefit benefit)
      : OpConversionPattern<triton::AtomicRMWOp>(converter, context, benefit),
        triton::TritonPtrScalarConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy = op.getResult().getType();
    // FIXME: lower to llvm.atom directly.
    if (resultTy.isa<RankedTensorType>())
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto memref = getMemRef(loc, op.getPtr(), resultType, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref, true, true);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    RankedTensorType originTensorTy =
        originTensor.getType().cast<RankedTensorType>();

    SmallVector<int64_t, 2> shape = {1, 1};
    auto val =
        rewriter.create<tensor::EmptyOp>(loc, shape, op.getVal().getType());
    Value valTensor = rewriter.create<tensor::InsertOp>(
        loc, op.getVal(), val, ValueRange({zero, zero}));
    Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    auto offset =
        rewriter.create<tensor::EmptyOp>(loc, shape, zeroI32.getType());
    Value offsetValue = rewriter.create<tensor::InsertOp>(
        loc, zeroI32, offset, ValueRange({zero, zero}));

    SmallVector<Value> atomicInputs{valTensor, offsetValue};

    // If mask exists, get mask from op.
    // If mask does not exist, set mask to 1.
    Value hasMask = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    if (op.getMask()) {
      hasMask = op.getMask();
    }
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, TypeRange{resultType}, hasMask, true);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    Value atomicResultInit = rewriter.create<tensor::EmptyOp>(
        loc, shape, originTensorTy.getElementType());

    auto maybeKind =
        getAtomicRMWType(op.getAtomicRmwOp(), originTensorTy.getElementType());
    if (!maybeKind)
      return failure();

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    SmallVector<Value> atomicInits{originTensor, atomicResultInit};
    Value atomicResult =
        rewriter
            .create<linalg_ext::GatherAtomicRMWOp>(
                loc, atomicInputs, atomicInits, *maybeKind, *maybeMemoryOrder)
            .getResult()[1];
    Value scalarRet =
        rewriter
            .create<tensor::ExtractOp>(loc, op.getResult().getType(),
                                       atomicResult, ValueRange({zero, zero}))
            .getResult();
    rewriter.create<scf::YieldOp>(loc, ValueRange(scalarRet));
    // Yield 0 if mask is false, align with GPU behaviour.
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    Value zeroConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(op.getResult().getType(), 0));
    rewriter.create<scf::YieldOp>(loc, ValueRange(zeroConst));
    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }
};

/// Contiguous atomic.
class TritonContiguousAtomicRmwOpConversion
    : public OpConversionPattern<triton::AtomicRMWOp>,
      public TritonPtrContiguousConversionBase {
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

public:
  TritonContiguousAtomicRmwOpConversion(TritonLinalgTypeConverter &converter,
                                        MLIRContext *context,
                                        DataFlowSolver &solver,
                                        PatternBenefit benefit)
      : OpConversionPattern<triton::AtomicRMWOp>(converter, context, benefit),
        TritonPtrContiguousConversionBase(solver) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultTy =
        op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy)
      return failure();

    auto loc = op.getLoc();
    auto ptrInfo = getPtrInfo(loc, op.getPtr(), op.getMask(), resultTy,
                              rewriter, nullptr, false);
    if (failed(ptrInfo)) {
      return failure();
    }

    if (!isConsecutive(ptrInfo->permutations)) {
      return failure();
    }
    for (const auto &dimInfo : llvm::enumerate(ptrInfo->dimInfos)) {
      if (dimInfo.value().isBroadcastDim()) {
        return failure();
      }
    }

    Value originalTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, ptrInfo->memref, true, true);

    // Create atomic_rmw here.
    // Init atomic output.
    Type resultEltType = resultTy.getElementType();
    Value atomicResultInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultTy.getShape(), resultEltType);
    auto maybeKind = getAtomicRMWType(op.getAtomicRmwOp(), resultEltType);
    if (!maybeKind)
      return failure();

    Value input = op.getVal();
    auto rank = atomicResultInit.getType().cast<ShapedType>().getRank();
    atomicResultInit = rewriter.create<tensor::ExtractSliceOp>(
        loc, atomicResultInit, ptrInfo->offsets, ptrInfo->sizes,
        SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));

    input = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, ptrInfo->offsets, ptrInfo->sizes,
        SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(1)));

    SmallVector<Value> atomicInits{originalTensor, atomicResultInit};
    // Get linalg_ext.atomic_rmw input operands.
    SmallVector<Value> atomicInputs{input};

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    Value sliceTensor =
        rewriter
            .create<triton::linalg_ext::AtomicRMWOp>(
                loc, atomicInputs, atomicInits, *maybeKind, *maybeMemoryOrder)
            ->getResult(1);
    // Pad value is set to 0 to align with GPU.
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, resultEltType, rewriter.getZeroAttr(resultEltType));
    sliceTensor = getPadOrInsertOpWithOther(
        loc, c0,
        RankedTensorType::get(resultTy.getShape(), resultTy.getElementType()),
        sliceTensor, ptrInfo->offsets, ptrInfo->sizes, rewriter);

    rewriter.replaceOp(op, sliceTensor);
    return success();
  }
};

/// Discrete atomic.
class TritonScatteredAtomicRMWOpConversion
    : public mlir::OpConversionPattern<triton::AtomicRMWOp>,
      public TritonPtrScatterConversionBase {
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

public:
  TritonScatteredAtomicRMWOpConversion(
      mlir::triton::TritonLinalgTypeConverter &converter,
      mlir::MLIRContext *context, mlir::DataFlowSolver &solver,
      mlir::PatternBenefit benefit)
      : OpConversionPattern<triton::AtomicRMWOp>(converter, context, benefit),
        TritonPtrScatterConversionBase(solver) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultTy)
      return failure();

    auto loc = op.getLoc();

    triton::PointerMetaInfoTracker tracker;
    if (failed(tracker.parse(op.getPtr(), loc, rewriter)))
      return failure();
    Value memref = getDynamicMemRef(loc, tracker.getBase(), resultTy, rewriter);
    Value originTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, memref, true, true);
    // Get value.
    Value valueTensor =
        triton::flattenValueToMatchGatherScatter(rewriter, op.getVal(), true);
    Value indices =
        triton::flattenValueToMatchGatherScatter(rewriter, tracker.getOffset());

    // Get linalg_ext.gather_atomic_rmw input operands.
    SmallVector<Value> atomicInputs{valueTensor, indices};

    // Get mask.
    if (op.getMask()) {
      auto atomicMask = triton::flattenValueToMatchGatherScatter(
          rewriter, op.getMask(), false);
      Value maskInit = rewriter.create<tensor::EmptyOp>(
          loc, atomicMask.getType().cast<RankedTensorType>().getShape(),
          rewriter.getI8Type());
      atomicMask = rewriter
                       .create<linalg::MapOp>(
                           loc, ValueRange{atomicMask}, maskInit,
                           [](OpBuilder &b, Location loc, ValueRange args) {
                             Value ret = b.create<arith::ExtSIOp>(
                                 loc, b.getI8Type(), args[0]);
                             b.create<linalg::YieldOp>(loc, ret);
                           })
                       .getResult()[0];
      atomicInputs.push_back(atomicMask);
    }

    // Init atomic output.
    Type resultEltType = resultTy.getElementType();
    Value atomicResultInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultTy.getShape(), resultEltType);
    atomicResultInit =
        triton::flattenValueToMatchGatherScatter(rewriter, atomicResultInit);

    auto maybeKind = getAtomicRMWType(op.getAtomicRmwOp(), resultEltType);
    if (!maybeKind)
      return failure();

    auto maybeMemoryOrder = getLinalgExtAtomicMemoryOrder(op.getSem());
    if (failed(maybeMemoryOrder))
      return failure();

    SmallVector<Value> atomicInits{originTensor, atomicResultInit};
    Value out =
        rewriter
            .create<linalg_ext::GatherAtomicRMWOp>(
                loc, atomicInputs, atomicInits, *maybeKind, *maybeMemoryOrder)
            ->getResult(1);

    // Reshape output to origin shape.
    out = triton::reshapeGatherScatterValueTo(out, resultTy, rewriter);
    rewriter.replaceOp(op, out);

    return success();
  }
};

void triton::populateTritonAtomicRmwToLinalgPatterns(
    RewritePatternSet &patterns, triton::TritonLinalgTypeConverter &converter,
    mlir::DataFlowSolver &solver) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonContiguousAtomicRmwOpConversion>(converter, context,
                                                      solver, 1);
  patterns.add<TritonScalarAtomicRMWOpConversion,
               TritonScatteredAtomicRMWOpConversion>(converter, context, solver,
                                                     0);
}
