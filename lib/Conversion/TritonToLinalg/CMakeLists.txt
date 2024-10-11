add_triton_library(TritonToLinalg
  AtomicCASConversion.cpp
  AtomicRmwConversion.cpp
  LoadStoreConversion.cpp
  TritonPointerConversion.cpp
  TritonToLinalg.cpp
  TypeConverter.cpp
  Utils.cpp

  DEPENDS
  TritonLinalgConverisonIncGen
  TritonInterfacesExtendTableGen

  LINK_LIBS PUBLIC
  LinalgExtDialectUtils
  TritonInterfaceExtend
  MLIRIR
)
