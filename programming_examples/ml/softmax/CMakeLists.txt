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
set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")
set(SIZE 262144 CACHE STRING "Number of elements")
set(CHANS 2 CACHE STRING "Number of channels")
set(N_ITERATIONS 20 CACHE STRING "Timed iterations")
set(N_WARMUP 10 CACHE STRING "Warm-up iterations")
# The host takes the NPU generation as a number, derived from the device
# (makefile-common: npu_num = 2 for npu2, else 1).
if(AIE_DEVICE STREQUAL "npu2")
  set(NPU_NUM 2)
else()
  set(NPU_NUM 1)
endif()
add_aie_design(TARGET ${currentTarget} PY softmax.py DEVICE ${AIE_DEVICE}
               ARGS -l ${SIZE} -ch ${CHANS})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS --npu ${NPU_NUM} --iters ${N_ITERATIONS}
                          --warmup ${N_WARMUP} --size ${SIZE})
