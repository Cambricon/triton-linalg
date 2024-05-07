add_triton_library(MathToLinalg
  MathToLinalg.cpp

  DEPENDS
  TritonLinalgConverisonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MathExtDialect
)
