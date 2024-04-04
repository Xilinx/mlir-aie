# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(AIE_LSP_LIBS
  ${dialect_libs}
  MLIRLspServerLib
)

add_llvm_tool(aie-lsp-server
  aie-lsp-server.cpp
)

target_link_libraries(aie-lsp-server PRIVATE ${AIE_LSP_LIBS})
