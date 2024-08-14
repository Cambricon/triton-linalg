add_triton_library(DialectUtils
  ArithUtils.cpp
  Conventions.cpp
  MemRefUtils.cpp
  ShapeUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  ArithExtDialect
)
