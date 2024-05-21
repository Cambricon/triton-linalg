add_triton_library(ArithTransforms
  ArithExtraCanonicalizer.cpp

  DEPENDS
  ArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
