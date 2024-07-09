add_triton_library(TritonLinalgPipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  ArithToLinalg
  MathToLinalg
  MLIRIR
  TritonToLinalg
  TritonTransformsExtend
)
