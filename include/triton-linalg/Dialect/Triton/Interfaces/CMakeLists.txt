set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS InferAxisInfoInterface.td)
mlir_tablegen(InferAxisInfoInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InferAxisInfoInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TritonInterfacesExtendTableGen)
