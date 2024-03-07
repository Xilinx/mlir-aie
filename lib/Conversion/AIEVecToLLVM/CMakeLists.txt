add_mlir_conversion_library(MLIRAIEVecToLLVM
  AIEVecToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  $(CMAKE_CURRENT_SRC_DIR)/../../../../include/aie/Conversion/AIEVecToLLVM

  DEPENDS
  MLIRAIEConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAIEVecDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
  MLIRXLLVMDialect
  )
