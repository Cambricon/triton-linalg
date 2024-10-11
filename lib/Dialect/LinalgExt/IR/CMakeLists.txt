add_triton_library(LinalgExtInterface
  LinalgExtInterface.cpp

  DEPENDS
  LinalgExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_triton_library(LinalgExtDialect
  LinalgExtOps.cpp
  LinalgExtDialect.cpp

  DEPENDS
  LinalgExtTableGen

  LINK_LIBS PUBLIC
  DialectUtils
  LinalgExtInterface
  MLIRIR
)
