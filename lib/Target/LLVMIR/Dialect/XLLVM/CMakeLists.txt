add_mlir_translation_library(MLIRXLLVMToLLVMIRTranslation
  XLLVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRXLLVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRXLLVMDialect
  )

