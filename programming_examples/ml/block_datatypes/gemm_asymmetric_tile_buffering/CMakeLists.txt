# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -Dsubdir:      Which configuration to build (config1|config2|config3).
#                Selects the host source and the problem sizes.
#
# This one CMakeLists is shared by every configuration, exactly as in the make
# flow: ../matrix_multiplication/makefile_common's ${targetname}.exe rule
# configures this directory with -DHOST_SRC/-DMATMUL_DIR/-DAIE_BUILD_DESIGN=OFF
# to produce only the host binary. The run_strix_cmake.lit tests configure it
# with -Dsubdir=<name> and AIE_BUILD_DESIGN left ON, so CMake additionally JITs
# the design and registers the ctest.

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../../common.cmake)

# --- Per-configuration defaults, mirroring each config's Makefile ------------
# m/k/n are baked into each config's in-tree microkernel and are not
# overridable; see the comments in the Makefiles.
if(subdir)
  if(subdir STREQUAL "config1")
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/gemm_atb_mixed_test.cpp")
    set(_M 4096)
    set(_K 4096)
    set(_N 2048)
    set(_m 128)
    set(_k 64)
    set(_n 128)
  elseif(subdir STREQUAL "config2")
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/gemm_atb_bfp_test.cpp")
    set(_M 3072)
    set(_K 4096)
    set(_N 1536)
    set(_m 192)
    set(_k 128)
    set(_n 96)
  elseif(subdir STREQUAL "config3")
    set(_host "${CMAKE_CURRENT_SOURCE_DIR}/gemm_atb_bfp_test.cpp")
    set(_M 4096)
    set(_K 4096)
    set(_N 2048)
    set(_m 128)
    set(_k 64)
    set(_n 128)
  else()
    message(FATAL_ERROR "Unknown -Dsubdir='${subdir}' (config1|config2|config3)")
  endif()

  if(NOT HOST_SRC)
    set(HOST_SRC "${_host}")
  endif()
endif()

if(NOT HOST_SRC)
  message(FATAL_ERROR
      "Pass -DHOST_SRC=<test.cpp> (the make flow) or -Dsubdir=<config1|config2|config3>")
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
# The CMake equivalent of ../matrix_multiplication/makefile_common: the
# ${aieargs} passed to <subdir>/n32_core.py and the `run:` recipe registered as
# a ctest. Guarded by AIE_BUILD_DESIGN, which makefile_common sets to OFF so
# that `make` does not JIT the design a second time through CMake.
if(AIE_BUILD_DESIGN AND subdir)

# BFP is aie2p-only, so there is no npu1 leg here.
set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu2)")

set(M ${_M} CACHE STRING "GEMM M")
set(K ${_K} CACHE STRING "GEMM K")
set(N ${_N} CACHE STRING "GEMM N")
set(TILE_M ${_m} CACHE STRING "AIE tile rows (Makefile `m`)")
set(TILE_K ${_k} CACHE STRING "AIE tile depth (Makefile `k`)")
set(TILE_N ${_n} CACHE STRING "AIE tile cols (Makefile `n`)")

add_aie_design(TARGET ${currentTarget} PY n32_core.py DEVICE ${AIE_DEVICE}
               PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
               ARGS -m ${TILE_M} -k ${TILE_K} -n ${TILE_N}
                    -M ${M} -K ${K} -N ${N})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS -M ${M} -K ${K} -N ${N}
                          -w ${TILE_M} -y ${TILE_K} -z ${TILE_N}
                          -v 2 --warmup 1 --iters 1)

endif() # AIE_BUILD_DESIGN AND subdir
