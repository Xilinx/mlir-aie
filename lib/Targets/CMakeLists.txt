add_mlir_library(AIETargets
  AIETargets.cpp
  AIETargetXAIEV1.cpp
  AIEFlowsToJSON.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  LINK_COMPONENTS
  Core
  IRReader
  Support
  TransformUtils
  )
target_link_libraries(AIETargets
  PUBLIC
  AIE
  MLIRTranslation
)
