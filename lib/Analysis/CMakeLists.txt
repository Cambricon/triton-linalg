add_triton_library(TritonLinalgAnalysis
  AxisInfoAnalysis.cpp

  LINK_LIBS PUBLIC
  TritonInterfaceExtend
  MLIRIR
)
