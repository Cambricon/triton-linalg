function(add_linalg_ods_yaml_gen yaml_ast_file output_file)
  set(YAML_AST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${yaml_ast_file})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  set(MLIR_LINALG_ODS_YAML_GEN_EXE mlir-linalg-ods-yaml-gen)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND ${MLIR_LINALG_ODS_YAML_GEN_EXE} ${YAML_AST_SOURCE} -o-ods-decl=${GEN_ODS_FILE} -o-impl=${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${YAML_AST_SOURCE}
    DEPENDS
    ${MLIR_LINALG_ODS_YAML_GEN_EXE}
    ${MLIR_LINALG_ODS_YAML_GEN_TARGET})
  add_custom_target(
    MLIR${output_file}YamlIncGen
    DEPENDS
    ${MLIR_LINALG_ODS_YAML_GEN_EXE}
    ${MLIR_LINALG_ODS_YAML_GEN_TARGET}
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
  set_target_properties(MLIR${output_file}YamlIncGen PROPERTIES FOLDER "MLIR/Tablegenning")
  list(APPEND LLVM_TARGET_DEPENDS ${GEN_ODS_FILE})
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} PARENT_SCOPE)
endfunction()

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

add_linalg_ods_yaml_gen(LinalgExtNamedStructuredOps.yaml LinalgExtNamedStructuredOps)

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
