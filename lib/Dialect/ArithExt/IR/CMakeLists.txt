add_triton_library(ArithExtDialect
  ArithExtOps.cpp
  ArithExtDialect.cpp

  DEPENDS
  ArithExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
