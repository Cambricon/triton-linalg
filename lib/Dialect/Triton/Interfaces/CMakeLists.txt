add_triton_library(TritonInterfaceExtend
  InferAxisInfoInterface.cpp

  DEPENDS
  TritonInterfacesExtendTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
