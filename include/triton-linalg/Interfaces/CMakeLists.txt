set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS InferResultTypeOpInterface.td)
mlir_tablegen(InferResultTypeOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InferResultTypeOpInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TritonLinalgInterfacesTableGen)
