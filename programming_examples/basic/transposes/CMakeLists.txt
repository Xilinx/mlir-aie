# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_include_directories (${currentTarget} PUBLIC ${XRT_INC_DIR})

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})

# Build the design and register the NPU run; -DAIE_DEVICE=npu2 targets Strix.
# Per-strategy defaults mirror the Makefile (and transposes.py's _DEFAULTS).
# They are set explicitly rather than left to the design script because the host
# testbench needs M/K/dtype-bytes too, and it cannot infer them from the strategy.
set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu|npu2)")
set(STRATEGY dma CACHE STRING "Transpose strategy (dma|dma_packet|shuffle|combined)")
set(_extra "")
if(STRATEGY STREQUAL "dma")
  set(_m 64)
  set(_k 64)
  set(_b 4)
elseif(STRATEGY STREQUAL "dma_packet")
  set(_m 64)
  set(_k 32)
  set(_b 4)
elseif(STRATEGY STREQUAL "shuffle")
  set(_m 16)
  set(_k 16)
  set(_b 1)
elseif(STRATEGY STREQUAL "combined")
  set(_m 128)
  set(_k 128)
  set(_b 4)
  set(_extra -m 32 -n 32 --ss 8)
else()
  message(FATAL_ERROR "Unknown STRATEGY '${STRATEGY}' (dma|dma_packet|shuffle|combined)")
endif()
set(M ${_m} CACHE STRING "Rows of the input matrix")
set(K ${_k} CACHE STRING "Columns of the input matrix")
set(DTYPE_BYTES ${_b} CACHE STRING "Element size in bytes (1|2|4)")
add_aie_design(TARGET ${currentTarget} PY transposes.py DEVICE ${AIE_DEVICE}
               ARGS -s ${STRATEGY} -M ${M} -K ${K} --dtype-bytes ${DTYPE_BYTES} ${_extra})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS -M ${M} -K ${K} --dtype-bytes ${DTYPE_BYTES})
