# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS
  Support
  )


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_tool(aie-translate
  aie-translate.cpp
  )

add_dependencies(aie-translate AIETargets)
llvm_update_compile_flags(aie-translate)
target_link_libraries(aie-translate
  PRIVATE
  ${dialect_libs}
  AIE
  AIETargets
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslation
  MLIRSupport
  MLIRTargetLLVMIR
  )
