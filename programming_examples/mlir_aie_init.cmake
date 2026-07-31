# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Encapsulates the pre-project() preamble shared by every example
# CMakeLists.txt under programming_examples/. Must be a macro (not a
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

    # Default the host compiler to MSVC. Otherwise CMake takes the first
    # clang++ on PATH, which in the IRON environment is llvm-aie's -- a
    # bare-metal AIE cross-compiler. It gets far enough to be selected and
    # then dies including MSVC's <intrin.h>, because its own xmmintrin.h
    # expects an mm_malloc.h that the bare-metal toolchain does not ship:
    #
    #   llvm-aie/lib/clang/21/include/xmmintrin.h:31:10:
    #     fatal error: 'mm_malloc.h' file not found
    #
    # Under WSL this is a no-op: `powershell.exe cmake` already selects MSVC,
    # and cl.exe is the right answer there too. An explicit -DCMAKE_CXX_COMPILER
    # still wins, which is how the vision examples pass gcc/g++ under WSL.
    if(NOT DEFINED CMAKE_C_COMPILER)
      set(CMAKE_C_COMPILER cl)
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      set(CMAKE_CXX_COMPILER cl)
    endif()

    # NOTE: /Zc:__cplusplus is NOT added here. It is MSVC-only, and this macro
    # runs before project(), so the compiler is not yet known -- MSVC is still
    # undefined at this point. common.cmake adds it after project() has done
    # language detection, guarded on MSVC. See the note there.
  endif()

  # Not "test": CTest reserves that target name once enable_testing() is in
  # effect, and common.cmake enables testing at directory scope. Every real
  # invocation passes -DTARGET_NAME anyway (build_host_exe and the run_cmake
  # lits both do), so this is only the fallback for a bare `cmake <example>`.
  set(TARGET_NAME test_exe CACHE STRING "Target to be built")
  set(ProjectName proj_${TARGET_NAME})
  set(currentTarget ${TARGET_NAME})
endmacro()
