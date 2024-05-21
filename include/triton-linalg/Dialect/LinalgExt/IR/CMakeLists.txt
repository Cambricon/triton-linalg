set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS LinalgExtInterface.td)
mlir_tablegen(LinalgExtInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(LinalgExtInterface.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS LinalgExtBase.td)
mlir_tablegen(LinalgExtOpsDialect.h.inc -gen-dialect-decls -dialect=linalg_ext)
mlir_tablegen(LinalgExtOpsDialect.cpp.inc -gen-dialect-defs -dialect=linalg_ext)

set(LLVM_TARGET_DEFINITIONS LinalgExtStructedOps.td)
mlir_tablegen(LinalgExtStructedOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgExtStructedOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS LinalgExtOps.td)
mlir_tablegen(LinalgExtOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgExtOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS LinalgExtEnums.td)
mlir_tablegen(LinalgExtEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgExtEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(LinalgExtTableGen)
