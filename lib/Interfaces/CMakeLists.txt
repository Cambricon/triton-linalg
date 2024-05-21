add_triton_library(TritonLinalgInterface
  InferResultTypeOpInterface.cpp

  DEPENDS
  TritonLinalgInterfacesTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
