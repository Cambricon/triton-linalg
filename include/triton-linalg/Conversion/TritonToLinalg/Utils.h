//===- Utils.h - ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_UTILS_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <stdint.h>

namespace mlir {
class Value;
class OpFoldResult;
class OpBuilder;
class MLIRContext;
namespace triton {
enum class CacheModifier : uint32_t;
enum class MemSemantic : uint32_t;
namespace linalg_ext {
enum class MemoryOrder : uint32_t;
} // namespace linalg_ext

} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {

/// Get pad op or insert_slice op.
Value getPadOrInsertOpWithOther(Location loc, Value other, Type otherType,
                                Value source, ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes,
                                OpBuilder &rewriter);

StringAttr getCacheModeAttr(MLIRContext *context, triton::CacheModifier mode);

/// Get atomic MemoryOrder.
FailureOr<triton::linalg_ext::MemoryOrder>
getLinalgExtAtomicMemoryOrder(triton::MemSemantic memSem);
} // namespace triton
} // namespace mlir
#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_UTILS_H
