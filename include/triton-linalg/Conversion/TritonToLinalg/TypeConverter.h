//===- TypeConverter.h - Convert Triton to Linalg dialect types -*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_TYPECONVERTER_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

class TritonLinalgTypeConverter : public TypeConverter {
public:
  TritonLinalgTypeConverter();
};

} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_TYPECONVERTER_H
