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
set(LENGTH 65536 CACHE STRING "Vector length")
set(CHANS 1 CACHE STRING "Number of channels")
set(OP add CACHE STRING "Elementwise op (add|mul)")
add_aie_design(TARGET ${currentTarget} PY eltwise.py DEVICE ${AIE_DEVICE} ARGS -l ${LENGTH} -ch ${CHANS} -o ${OP})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget} RUN_ARGS --op ${OP})
