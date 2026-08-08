# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CHANNELS 64 CACHE STRING "channel count")
set(SEQ_LEN 128 CACHE STRING "sequence length per channel")
set(KERNEL_SIZE 9 CACHE STRING "depthwise conv1d tap count")
set(BIAS 1 CACHE STRING "1 if the kernel adds a per-channel bias, else 0")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
    CHANNELS=${CHANNELS}
    SEQ_LEN=${SEQ_LEN}
    KERNEL_SIZE=${KERNEL_SIZE}
    BIAS=${BIAS}
)

target_include_directories (${currentTarget} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})
