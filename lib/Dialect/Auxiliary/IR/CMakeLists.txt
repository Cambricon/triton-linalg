add_triton_library(AuxiliaryDialect
  AuxiliaryDialect.cpp

  DEPENDS
  AuxiliaryTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
