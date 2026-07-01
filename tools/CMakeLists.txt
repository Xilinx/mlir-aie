#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_subdirectory(aie-opt)
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_subdirectory(aie-reset)
endif()
if(AIE_BUILD_LSP_SERVER)
  add_subdirectory(aie-lsp-server)
endif()
add_subdirectory(aie-translate)
if(AIE_BUILD_VISUALIZE)
  add_subdirectory(aie-visualize)
endif()
add_subdirectory(aiecc)
add_subdirectory(bootgen)
if(AIE_BUILD_CHESS_CLANG)
  add_subdirectory(chess-clang)
endif()
