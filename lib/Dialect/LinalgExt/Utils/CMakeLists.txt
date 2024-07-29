add_triton_library(LinalgExtDialectUtils
  Utils.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
