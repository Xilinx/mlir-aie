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
# bootgen must be configured before aiecc so the `bootgen-lib` target exists
# when aiecc's CMakeLists checks `if(TARGET bootgen-lib)` to enable in-process
# PDI generation via the bootgen C API.
add_subdirectory(bootgen)
add_subdirectory(aiecc)
if(AIE_BUILD_CHESS_CLANG)
  add_subdirectory(chess-clang)
endif()
