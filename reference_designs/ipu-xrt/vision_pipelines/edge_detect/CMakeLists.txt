# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.

# parameters
# -DBOOST_ROOT: Path to Boost install
# -DOpenCV_DIR: Path to OpenCV install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(BOOST_ROOT C:/Technical/thirdParty/boost_1_83_0 CACHE STRING "Path to Boost install")
set(OpenCV_DIR C:/Technical/thirdParty/opencv/build CACHE STRING "Path to OpenCV install")
set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT cloned repo")
set(XRT_LIB_DIR C:/Technical/xrtIPUfromDLL CACHE STRING "Path to xrt_coreutil.lib")

SET (ProjectName edgeDetect)
SET (currentTarget edgeDetect)

project(${ProjectName})

# Find packages
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
message("opencv library paht: ${OpenCV_LIB_PATH}")
message("opencv libs: ${OpenCV_LIBS}")


add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/OpenCVUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/xrtUtils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC DISABLE_ABI_CHECK=1)

target_include_directories (${currentTarget} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${XRT_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${OpenCV_LIB_PATH}
    ${Boost_LIBRARY_DIRS}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
    ${OpenCV_LIBS}
)