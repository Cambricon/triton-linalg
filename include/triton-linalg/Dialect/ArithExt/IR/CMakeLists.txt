set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS ArithExtBase.td)
mlir_tablegen(ArithExtOpsDialect.h.inc -gen-dialect-decls -dialect=arith_ext)
mlir_tablegen(ArithExtOpsDialect.cpp.inc -gen-dialect-defs -dialect=arith_ext)

set(LLVM_TARGET_DEFINITIONS ArithExtOps.td)
mlir_tablegen(ArithExtOps.h.inc -gen-op-decls)
mlir_tablegen(ArithExtOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(ArithExtTableGen)
