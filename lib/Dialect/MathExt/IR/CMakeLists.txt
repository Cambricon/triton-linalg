add_triton_library(MathExtDialect
  MathExtOps.cpp
  MathExtDialect.cpp

  DEPENDS
  MathExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
