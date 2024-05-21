add_triton_library(AuxiliaryTransforms
  AuxOpTilingInterface.cpp

  LINK_LIBS PUBLIC
  AuxiliaryDialect
  MLIRIR
)
