# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

add_llvm_executable(aie-translate aie-translate.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS
  ${dialect_libs}
  AIE
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
)

add_dependencies(aie-translate AIETargets)
llvm_update_compile_flags(aie-translate)
target_link_libraries(aie-translate PRIVATE MLIRIR MLIRTranslation ${LIBS} LLVMSupport)
