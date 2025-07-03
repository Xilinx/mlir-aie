# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Xilinx Inc.

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(WSL NAMES powershell.exe)

if (NOT WSL)

    set(CMAKE_C_COMPILER gcc-13)
    set(CMAKE_CXX_COMPILER g++-13)
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")
else()
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")
endif ()

set(IN1_SIZE 4096 CACHE STRING "in1 buffer size")
set(OUT_SIZE 4096 CACHE STRING "out buffer size")
set(DTYPE i32 CACHE STRING "data type")
set(TARGET_NAME test CACHE STRING "Target to be built")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

if ( WSL )
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    add_compile_options(/Zc:__cplusplus)
endif ()

project(${ProjectName})


add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    OUT_SIZE=${OUT_SIZE}
    DTYPE=${DTYPE}
    DISABLE_ABI_CHECK=1
        )

target_include_directories (${currentTarget} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)