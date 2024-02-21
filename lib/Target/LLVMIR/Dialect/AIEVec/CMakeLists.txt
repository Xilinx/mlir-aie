add_mlir_translation_library(MLIRAIEVecToLLVMIRTranslation
  AIEVecToLLVMIRTranslation.cpp

  DEPENDS
  MLIRAIEVecConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAIEVecDialect
  MLIRAIEVecUtils
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )

