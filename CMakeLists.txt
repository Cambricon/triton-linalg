set(TRITON_LINALG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_LINALG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

add_triton_plugin(TritonLinalg ${CMAKE_CURRENT_SOURCE_DIR}/triton_linalg.cc LINK_LIBS AuxiliaryDialect LinalgExtDialect)
