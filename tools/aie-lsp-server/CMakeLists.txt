# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(AIE_LSP_LIBS
  ${dialect_libs}
  MLIRLspServerLib
  MLIRRegisterAllDialects
)

add_llvm_tool(aie-lsp-server
  aie-lsp-server.cpp
)
install(TARGETS aie-lsp-server
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-lsp-server)

target_link_libraries(aie-lsp-server PRIVATE ${AIE_LSP_LIBS})
