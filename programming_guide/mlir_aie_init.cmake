# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# Encapsulates the pre-project() preamble shared by every example
# CMakeLists.txt under programming_guide/. Must be a macro (not a
# function) so that CMAKE_*_COMPILER settings and the cache/local
# variables it sets are visible in the caller's scope before project()
# runs.
#
# Intended usage in a template:
#
#   cmake_minimum_required(VERSION 3.30)
#   include(<path-to>/mlir_aie_init.cmake)
#   mlir_aie_init_example()       # WSL + compilers + ProjectName/currentTarget
#   project(${ProjectName})       # MUST be a literal call in the top-level
#                                 # CMakeLists.txt (CMake requirement)
#   include(<path-to>/common.cmake)
#
# Why this exists: common.cmake calls find_package(XRT), which loads
# xrt-targets.cmake and runs add_library(... SHARED IMPORTED). That
# requires project() to have been called first. Centralising the
# preamble here keeps the ordering rule in one place rather than
# duplicated across ~50 templates. See Xilinx/mlir-aie#3048.
#
# Templates that need to override anything (extra cache vars, a
# different default TARGET_NAME, custom XRT paths, etc.) should set it
# BEFORE calling this macro. The CACHE-form set() calls below are no-ops
# if the value is already in the cache.

macro(mlir_aie_init_example)
  find_program(WSL NAMES powershell.exe)

  if(NOT WSL)
    if(NOT DEFINED CMAKE_C_COMPILER)
      set(CMAKE_C_COMPILER gcc-13)
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      set(CMAKE_CXX_COMPILER g++-13)
    endif()
  else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    add_compile_options(/Zc:__cplusplus)
  endif()

  set(TARGET_NAME test CACHE STRING "Target to be built")
  set(ProjectName proj_${TARGET_NAME})
  set(currentTarget ${TARGET_NAME})
endmacro()
