#include "./RegisterTritonLinalgDialects.h"
#include "triton-linalg/Pipelines/Pipelines.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::mlir::registerAllExtensions(registry);
  ::mlir::registerAllPasses();
  registerTritonLinalgDialects(registry);
  registerTritonLinalgPasses();
  ::mlir::triton::registerTritonToLinalgPass();
  ::mlir::triton::registerTritonLinalgPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Linalg test driver\n", registry));
}
