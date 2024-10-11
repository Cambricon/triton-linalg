set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS MathExtBase.td)
mlir_tablegen(MathExtOpsDialect.h.inc -gen-dialect-decls -dialect=math_ext)
mlir_tablegen(MathExtOpsDialect.cpp.inc -gen-dialect-defs -dialect=math_ext)

set(LLVM_TARGET_DEFINITIONS MathExtOps.td)
mlir_tablegen(MathExtOps.h.inc -gen-op-decls)
mlir_tablegen(MathExtOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MathExtTableGen)
