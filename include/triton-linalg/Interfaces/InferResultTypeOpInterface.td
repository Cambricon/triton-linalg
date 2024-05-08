//===- InferResultTypeOpInterface.td - infer type interface-*- tablegen -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can infer result types by input
// values and attributes.
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_INTERFACES_INFERRESULTTYPEOPINTERFACE_TD
#define TRITON_LINALG_INTERFACES_INFERRESULTTYPEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferResultTypeOpInterface : OpInterface<"InferResultTypeOpInterface"> {
  let description = [{
    Op with this interface can infer result type by input values and attributes.
  }];
  let cppNamespace = "::mlir::triton";
  let methods = [
      InterfaceMethod<
      /*desc=*/[{
        This method infers result type on operation, return the infered type.
      }],
      /*retType=*/"llvm::SmallVector<::mlir::Type>",
      /*methodName=*/"inferResultTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
      >,
      InterfaceMethod<
      /*desc=*/[{
        This method infers result type on operation, update result value
        with inferred type inplace.
      }],
      /*retType=*/"void",
      /*methodName=*/"updateResultType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto &&resultTypes = cast<InferResultTypeOpInterface>(
            $_op.getOperation()).inferResultTypes();
        for (const auto &en : llvm::enumerate($_op.getOperation()->getResults())) {
          en.value().setType(resultTypes[en.index()]);
        }
      }]
      >
  ];
  let extraClassDeclaration = [{
  }];
}

#endif // TRITON_LINALG_INTERFACES_INFERRESULTTYPEOPINTERFACE_TD
