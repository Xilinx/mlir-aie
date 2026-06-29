#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

if(NOT DEFINED CMAKE_CROSSCOMPILE)
  set(SCRIPT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
  # These files are only useful on host x86 compiles.
  add_custom_target(xchesscc_wrapper
          COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/xchesscc_wrapper
          ${PROJECT_BINARY_DIR}/bin/xchesscc_wrapper
          SOURCES
          ${CMAKE_CURRENT_SOURCE_DIR}/xchesscc_wrapper)
  install(PROGRAMS xchesscc_wrapper DESTINATION ${SCRIPT_INSTALL_PATH})
endif()