add_triton_library(TritonToTensor
  TritonToTensor.cpp

  DEPENDS
  TritonLinalgConverisonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
