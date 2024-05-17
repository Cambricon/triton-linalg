//===- InferAxisInfoInterface.td -infer axis info interface-*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for axis info analysis.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_TD
#define TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferAxisInfoInterface : OpInterface<"InferAxisInfoInterface"> {
  let description = [{
    Allows operation to participate in axis info analysis by providing a method
    that allows them to specify info such as contiguity, divisibility, etc.
    bound to the axis when their input info is given.
  }];
  let cppNamespace = "::mlir::triton";
  let methods = [
    InterfaceMethod<[{
      Infer the axis info on the results of this op given the axis info on its
      arguments.

      `argInfos` contains one `AxisInfo` for each argument to the op in ODS order.
    }],
    "void", "inferAxisInfos", (ins
      "::llvm::ArrayRef<::mlir::triton::AxisInfoExt>":$argInfos,
      "::mlir::triton::SetAxisInfoFn":$setResultAxisInfo)
  >];
}

#endif // TRITON_LINALG_DIALECT_TRITON_INTERFACES_INFERAXISINFOINTERFACE_TD
