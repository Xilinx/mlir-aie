# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

add_llvm_executable(aie-translate aie-translate.cpp)
llvm_update_compile_flags(aie-translate)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
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
