# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -Dsubdir:      Which parameterization to build (single_core, whole_array,
#                in_core_shuffle, ...). Selects the design .py, the host
#                source, and the problem sizes.
#
# This one CMakeLists is shared by every parameterization, exactly as in the
# make flow: makefile_common's ${targetname}.exe rule configures this directory
# with -DHOST_SRC/-DMATMUL_DIR/-DAIE_BUILD_DESIGN=OFF to produce only the host
# binary. The run_strix_cmake.lit tests configure it with -Dsubdir=<name> and
# AIE_BUILD_DESIGN left ON, so CMake additionally JITs the design and registers
# the ctest.

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../../common.cmake)

# --- Per-parameterization defaults, mirroring each subdir's Makefile ---------
# m/k/n and the -v/--warmup/--iters runargs come from makefile_common; only the
# values a subdir's own Makefile overrides are listed here.
if(subdir)
  set(_py single_core.py)
  set(_host "${CMAKE_CURRENT_SOURCE_DIR}/bfp_test.cpp")
  set(_M 512)
  set(_K 512)
  set(_N 512)
  set(_tile_k 64)
  set(_extra_aieargs "")

  if(subdir STREQUAL "single_core")
  elseif(subdir STREQUAL "single_core_mixed")
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/mixed_test.cpp")
  elseif(subdir STREQUAL "single_core_no_tiling")
    set(_M 64)
    set(_K 128)
    set(_N 64)
    set(_tile_k 128)
  elseif(subdir STREQUAL "single_core_no_tiling_mixed")
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/mixed_test.cpp")
    set(_M 64)
    set(_K 64)
    set(_N 64)
  elseif(subdir STREQUAL "in_core_shuffle")
    # The only parameterization with its own host source rather than a shared one.
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/in_core_shuffle/test.cpp")
    set(_M 64)
    set(_K 64)
    set(_N 64)
  elseif(subdir STREQUAL "whole_array")
    set(_py whole_array.py)
    set(_extra_aieargs --n-aie-cols 4)
  elseif(subdir STREQUAL "whole_array_mixed")
    # No --n-aie-cols: this Makefile leaves whole_array.py's own default.
    set(_py whole_array.py)
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/mixed_test.cpp")
  elseif(subdir STREQUAL "whole_array_shuffle")
    set(_py whole_array.py)
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/whole_array_shuffle/bfp_test.cpp")
    set(_extra_aieargs --n-aie-cols 4)
  else()
    message(FATAL_ERROR "Unknown -Dsubdir='${subdir}'")
  endif()

  if(NOT HOST_SRC)
    set(HOST_SRC "${_host}")
  endif()
endif()

if(NOT HOST_SRC)
  message(FATAL_ERROR
      "Pass -DHOST_SRC=<test.cpp> (the make flow) or -Dsubdir=<parameterization>")
endif()
if(NOT MATMUL_DIR)
  set(MATMUL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../basic/matrix_multiplication")
endif()

add_executable(${currentTarget}
    ${HOST_SRC}
)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
    ${MATMUL_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_strix_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of makefile_common: the ${aieargs} passed to the design
# .py and the `run:` recipe registered as a ctest. Guarded by AIE_BUILD_DESIGN,
# which makefile_common sets to OFF -- without that, `make` would JIT the design
# a second time through CMake, which is what broke this directory before.
if(AIE_BUILD_DESIGN AND subdir)

# BFP is aie2p-only, so unlike basic/matrix_multiplication there is no npu1 leg.
set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu2)")

set(M ${_M} CACHE STRING "GEMM M")
set(K ${_K} CACHE STRING "GEMM K")
set(N ${_N} CACHE STRING "GEMM N")
set(TILE_M 64 CACHE STRING "AIE tile rows (Makefile `m`)")
set(TILE_K ${_tile_k} CACHE STRING "AIE tile depth (Makefile `k`)")
set(TILE_N 64 CACHE STRING "AIE tile cols (Makefile `n`)")

add_aie_design(TARGET ${currentTarget} PY ${_py} DEVICE ${AIE_DEVICE}
               PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
               ARGS -m ${TILE_M} -k ${TILE_K} -n ${TILE_N}
                    -M ${M} -K ${K} -N ${N} ${_extra_aieargs})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS -M ${M} -K ${K} -N ${N}
                          -w ${TILE_M} -y ${TILE_K} -z ${TILE_N}
                          -v 2 --warmup 1 --iters 1)

endif() # AIE_BUILD_DESIGN AND subdir
