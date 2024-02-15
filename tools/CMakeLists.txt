#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(aie-opt)
if(NOT WIN32)
  add_subdirectory(aie-reset)
endif()
add_subdirectory(aie-translate)
add_subdirectory(bootgen)
add_subdirectory(chess-clang)
add_subdirectory(aie2xclbin)
