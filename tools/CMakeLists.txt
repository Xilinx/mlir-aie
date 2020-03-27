# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_executable(aie-opt aie-opt.cpp)
llvm_update_compile_flags(aie-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopOpsTransforms
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  MLIROptLib
  AIE
  LLVMSupport
  LLVMCore
  LLVMAsmParser
  )
target_link_libraries(aie-opt ${LIBS})


add_executable(aie-translate aie-translate.cpp)
llvm_update_compile_flags(aie-translate)

set(LIBS
  ${dialect_libs}
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
)

add_dependencies(aie-translate AIETargets)
whole_archive_link(aie-translate AIETargets)
target_link_libraries(aie-translate PRIVATE MLIRIR MLIRTranslateClParser ${LIBS} LLVMSupport)
