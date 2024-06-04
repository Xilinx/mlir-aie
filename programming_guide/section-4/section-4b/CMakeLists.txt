# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# parameters
# -DBOOST_ROOT: Path to Boost install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

find_program(WSL NAMES powershell.exe)

if (NOT WSL)
    set(BOOST_ROOT /usr/include/boost CACHE STRING "Path to Boost install")
    set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil.lib")
else()
    set(BOOST_ROOT C:/Technical/thirdParty/boost_1_83_0 CACHE STRING "Path to Boost install")
    set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
    set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")
endif()

set(TARGET_NAME test CACHE STRING "Target to be built")

SET (ProjectName ${TARGET_NAME})
SET (currentTarget ${TARGET_NAME})

if ( WSL )
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
endif ()

project(${ProjectName})

# Find packages
find_package(Boost REQUIRED)

add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories (${currentTarget} PUBLIC 
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

if (NOT WSL)
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
        boost_program_options
        boost_filesystem
    )
else()
    target_link_libraries(${currentTarget} PUBLIC
        xrt_coreutil
    )
endif()