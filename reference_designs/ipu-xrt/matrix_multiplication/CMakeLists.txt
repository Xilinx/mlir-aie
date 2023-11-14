# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.

# parameters
# -DBOOST_ROOT: Path to Boost install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(BOOST_ROOT C:/Technical/thirdParty/boost_1_83_0 CACHE STRING "Path to Boost install")
set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
set(XRT_LIB_DIR C:/Technical/xrtIPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")

SET (ProjectName matrixMultiplication)
SET (currentTarget matrixMultiplication)

project(${ProjectName})

# Find packages
find_package(Boost REQUIRED)

add_executable(${currentTarget}
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories (${currentTarget} PUBLIC 
    ${XRT_INC_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${Boost_LIBRARY_DIRS}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)
