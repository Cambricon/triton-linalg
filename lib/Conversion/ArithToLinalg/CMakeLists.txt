add_triton_library(ArithToLinalg
  ArithToLinalg.cpp

  DEPENDS
  TritonLinalgConverisonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MathExtDialect
)
