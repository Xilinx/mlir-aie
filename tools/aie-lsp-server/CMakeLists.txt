get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(AIE_LSP_LIBS
  ${dialect_libs}
  MLIRLspServerLib
)

add_llvm_tool(aie-lsp-server
  aie-lsp-server.cpp
#  DEPENDS
#  ${AIE_LSP_LIBS}
)

target_link_libraries(aie-lsp-server PRIVATE ${AIE_LSP_LIBS})
