add_mlir_library(AIETargets
  AIETargets.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  )
target_link_libraries(AIETargets
  PUBLIC
  AIE
  LLVMCore
  LLVMIRReader
  LLVMSupport
  LLVMTransformUtils
  MLIRTranslation
)
