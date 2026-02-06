# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED TARGET_NAME)
  set(TARGET_NAME "chaining_channels")
endif()

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib)

add_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime_lib/test_lib/test_utils.cpp
    test.cpp
)

find_package(XRT REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE XRT::xrt_coreutil)
