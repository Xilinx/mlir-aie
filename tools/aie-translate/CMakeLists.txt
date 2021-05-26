# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS
  Support
  )


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
message("Translations: ${translation_libs}")

add_llvm_tool(aie-translate
  aie-translate.cpp
  )
install(TARGETS aie-translate
  EXPORT AIETargets
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aie-translate)

add_dependencies(aie-translate AIETargets)
llvm_update_compile_flags(aie-translate)
target_link_libraries(aie-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  AIE
  AIETargets
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslation
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  )
