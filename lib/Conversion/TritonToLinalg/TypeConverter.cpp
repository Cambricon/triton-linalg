//===- TypeConverter.cpp - Triton to Linalg dialect types -------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "triton-linalg/Conversion/TritonToLinalg/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

// Maps each triton pointer type to an int64.
static std::optional<Type> convertPointerToInt64(Type type) {
  return llvm::TypeSwitch<Type, std::optional<Type>>(type)
      .Case<triton::PointerType>(
          [](triton::PointerType tritonPtrType) -> std::optional<Type> {
            return IntegerType::get(tritonPtrType.getContext(), 64);
          })
      .Case<RankedTensorType>(
          [](RankedTensorType tensorType) -> std::optional<Type> {
            auto eleTy = tensorType.getElementType();
            auto pointeeType = convertPointerToInt64(eleTy);
            if (pointeeType) {
              return RankedTensorType::get(tensorType.getShape(),
                                           pointeeType.value());
            }
            return std::nullopt;
          })
      .Default([](Type type) { return type; });
}

TritonLinalgTypeConverter::TritonLinalgTypeConverter() {
  addConversion(convertPointerToInt64);

  // Add generic source and target materializations to handle cases where
  // int64 as triton.ptr types persist after an conversion.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult(0));
  };
  addSourceMaterialization(addUnrealizedCast);
  addTargetMaterialization(addUnrealizedCast);
  addArgumentMaterialization(addUnrealizedCast);
}
