# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(dialect_libs = ${dialect_libs})
set(AIE_LSP_LIBS
  ${dialect_libs}
  $<IF:$<BOOL:${CLANGIR_MLIR_FRONTEND}>,
    MLIRCIR
  ,>
  MLIRLspServerLib
)

add_llvm_tool(aie-lsp-server
  aie-lsp-server.cpp
)

install(TARGETS aie-lsp-server
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-lsp-server)

target_link_libraries(aie-lsp-server PRIVATE ${AIE_LSP_LIBS})
if (CLANGIR_MLIR_FRONTEND)
  target_compile_definitions(aie-lsp-server PRIVATE CLANGIR_MLIR_FRONTEND)
endif()
