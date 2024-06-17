//===- PointerInfo.h - Triton pointer info ----------------------*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERINFO_H
#define TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERINFO_H
#include <stdint.h>

namespace mlir {
namespace triton {

/// Structure info representation for pointer in triton.
class PtrInfo {
public:
  PtrInfo() = delete;
  PtrInfo(Value ptr, ArrayRef<Value> offsets)
      : pointer(ptr), tensorPtrOffsets(offsets) {}

  PtrInfo(Value ptr, ArrayRef<Value> sizes, ArrayRef<Value> strides,
          ArrayRef<Value> offsets, ArrayRef<int32_t> order)
      : pointer(ptr), tensorPtrSizes(sizes), tensorPtrStrides(strides),
        tensorPtrOffsets(offsets), tensorPtrOrder(order) {}

  PtrInfo(Value ptr, Value offset) : pointer(ptr) {
    tensorPtrOffsets.push_back(offset);
    isRawPtrInfo = true;
  }

  Value ptr() const { return pointer; }

  ArrayRef<Value> offsets() const { return tensorPtrOffsets; }
  Value offset(unsigned idx) const { return tensorPtrOffsets[idx]; }
  Value offset() const { return tensorPtrOffsets[0]; }
  void setOffsets(ValueRange vals) {
    for (unsigned i = 0; i < vals.size(); i++) {
      tensorPtrOffsets[i] = vals[i];
    }
  }
  unsigned offsetSize() { return tensorPtrOffsets.size(); }

  bool isBlockPtr() const { return !isRawPtrInfo; }

  ArrayRef<Value> sizes() const { return tensorPtrSizes; }
  ArrayRef<Value> strides() const { return tensorPtrStrides; }
  ArrayRef<int32_t> order() const { return tensorPtrOrder; }

private:
  bool isRawPtrInfo{false};
  Value pointer;
  // Basic info for reconstruction of MakeTensorPtrOp.
  SmallVector<Value> tensorPtrSizes;
  SmallVector<Value> tensorPtrStrides;
  SmallVector<Value> tensorPtrOffsets;
  SmallVector<int32_t> tensorPtrOrder;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_DIALECT_TRITON_UTILS_POINTERINFO_H
