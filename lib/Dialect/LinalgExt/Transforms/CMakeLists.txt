add_triton_library(LinalgExtTransforms
  LinalgExtOpTilingInterface.cpp
  TilingInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  LinalgExtDialect
  MLIRIR
)
