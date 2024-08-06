#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(aie-opt)
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_subdirectory(aie-reset)
endif()
add_subdirectory(aie-lsp-server)
add_subdirectory(aie-translate)
add_subdirectory(aie-visualize)
add_subdirectory(bootgen)
add_subdirectory(chess-clang)
