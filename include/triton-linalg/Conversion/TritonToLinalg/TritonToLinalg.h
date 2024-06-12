//===- TritonToLinalg.h - Triton to Linalg dialect convension ---*- C++ -*-===//
//
// Copyright (C) [2022-2025] by Cambricon.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
#define TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class ConversionTarget;
class DataFlowSolver;
namespace triton {
class TritonLinalgTypeConverter;

class TritonToLinalgPass
    : public PassWrapper<TritonToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonToLinalgPass)
  ::llvm::StringRef getArgument() const override {
    return "convert-triton-to-linalg";
  }

  ::llvm::StringRef getDescription() const override {
    return "Convert the operations from the Triton dialect into the Linalg "
           "dialect";
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override;

  void runOnOperation() final;

protected:
  virtual void populatePatterns(RewritePatternSet &patterns,
                                TritonLinalgTypeConverter &converter,
                                ConversionTarget &target,
                                mlir::DataFlowSolver &solver);
};

/// Create a pass to convert a subset of Triton ops to Linalg.
std::unique_ptr<mlir::Pass> createTritonToLinalgPass();

inline void registerTritonToLinalgPass() {
  PassRegistration<TritonToLinalgPass>();
}
} // namespace triton
} // namespace mlir

#endif // TRITON_LINALG_CONVERSION_TRITONTOLINALG_TRITONTOLINALG_H
