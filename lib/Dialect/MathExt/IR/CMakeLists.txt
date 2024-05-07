add_triton_library(MathExtDialect
  MathOps.cpp
  MathDialect.cpp

  DEPENDS
  MathExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
