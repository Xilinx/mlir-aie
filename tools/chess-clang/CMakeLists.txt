#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

if(NOT DEFINED CMAKE_CROSSCOMPILE)
  set(SCRIPT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
  # These files are only useful on host x86 compiles.
  install(PROGRAMS xchesscc_wrapper DESTINATION ${PROJECT_BINARY_DIR}/bin)
  install(PROGRAMS xchesscc_wrapper DESTINATION ${SCRIPT_INSTALL_PATH})
endif()