
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_executable(aie-lsp aie-lsp.cpp)
llvm_update_compile_flags(aie-lsp)
add_dependencies(aie-lsp 
  MLIRADFIncGen
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEPassIncGen
  MLIRAIEVecPassIncGen
)
install(TARGETS aie-lsp
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-lsp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  ADF
  AIE
  AIETransforms
  AIEUtils
  MLIRAIEVec
  MLIRAIEVecTransforms
  MLIRAIEVecToLLVM
  MLIRLspServerLib
  )
target_link_libraries(aie-lsp PRIVATE ${LIBS})
