set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS AuxiliaryDialect.td)
mlir_tablegen(AuxiliaryOpsDialect.h.inc -gen-dialect-decls -dialect=aux)
mlir_tablegen(AuxiliaryOpsDialect.cpp.inc -gen-dialect-defs -dialect=aux)

set(LLVM_TARGET_DEFINITIONS AuxiliaryTypes.td)
mlir_tablegen(AuxiliaryOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AuxiliaryOpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS AuxiliaryOps.td)
mlir_tablegen(AuxiliaryOps.h.inc -gen-op-decls)
mlir_tablegen(AuxiliaryOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(AuxiliaryTableGen)
