add_mlir_dialect_library(AIE
  AIEDialect.cpp
  AIEFindFlows.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  )

target_link_libraries(AIE
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
