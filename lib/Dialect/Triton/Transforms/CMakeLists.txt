add_triton_library(TritonTransformsExtend
  CanonicalizeTriton.cpp
  ExtractMoveBackward.cpp
  InferAxisInfoInterfaceImpl.cpp
  PointerStrengthReduction.cpp
  WrapFuncBodyWithSingleBlock.cpp

  DEPENDS
  TritonTransformsExtendIncGen

  LINK_LIBS PUBLIC
  LinalgExtDialect
  TritonDialectUtils
  TritonLinalgUtils
  TritonInterfaceExtend
  MLIRIR
)
