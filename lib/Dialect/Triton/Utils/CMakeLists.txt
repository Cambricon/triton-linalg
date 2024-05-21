add_triton_library(TritonDialectUtils
  MaskTracker.cpp
  PointerMetaInfoTracker.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
