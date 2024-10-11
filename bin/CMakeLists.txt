get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-linalg-opt triton-linalg-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-linalg-opt)
target_link_libraries(triton-linalg-opt PRIVATE
  AuxiliaryTransforms
  LinalgExtTransforms
  TritonLinalgAnalysis
  TritonLinalgPipelines

  TritonIR
  TritonGPUIR
  TritonAnalysis
  ${dialect_libs}
  ${conversion_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRFuncAllExtensions
)

mlir_check_all_link_libraries(triton-linalg-opt)
