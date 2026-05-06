# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# event_trace uses an example-specific TARGET_NAME and a hard-wired XRT
# install path. Setting these BEFORE mlir_aie_init_example() / common.cmake
# means the macro's CACHE-form set(TARGET_NAME test ...) is a no-op and
# common.cmake sees these XRT paths instead of probing.
set(TARGET_NAME mlir_trace_example CACHE STRING "Target to be built")
find_program(WSL NAMES powershell.exe)
if(NOT WSL)
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")
else()
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")
endif()

set(IN1_SIZE 16384 CACHE STRING "in1 buffer size")
set(IN2_SIZE 4 CACHE STRING "in2 buffer size")
set(OUT_SIZE 16384 CACHE STRING "out buffer size")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    IN2_SIZE=${IN2_SIZE}
    OUT_SIZE=${OUT_SIZE}
)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})
