#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#

# Declare that CMakeLists files in this directory and below should not be subject to rebuilds
cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

find_program(WSL NAMES powershell.exe)

if (NOT WSL)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
else()
  set(CMAKE_C_COMPILER gcc.exe)
  set(CMAKE_CXX_COMPILER g++.exe)
endif()

project(passthrough_kernel_cpp_gen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find LLVM for command line support
find_package(LLVM REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Passthrough kernel generator executable
add_executable(passthrough_kernel_gen passthrough_kernel_placed.cpp)

target_link_libraries(passthrough_kernel_gen
  PRIVATE
    LLVMSupport
)

# Test executable (for running the generated kernel)
if(NOT DEFINED TARGET_NAME)
  set(TARGET_NAME passthrough_kernel_cpp)
endif()

if(NOT DEFINED IN1_SIZE)
  set(IN1_SIZE 4096)
endif()

if(NOT DEFINED OUT_SIZE)
  set(OUT_SIZE 4096)
endif()

set(XRT_INCLUDE "/opt/xilinx/xrt/include")
set(XRT_LIB "/opt/xilinx/xrt/lib")

file(GLOB TESTCPP "test.cpp")

add_executable(${TARGET_NAME} ${TESTCPP})

target_compile_definitions(${TARGET_NAME} PRIVATE
  IN1_SIZE=${IN1_SIZE} OUT_SIZE=${OUT_SIZE}
  XAIE_DEBUG
)

target_include_directories(${TARGET_NAME} PRIVATE ${XRT_INCLUDE})
target_link_directories(${TARGET_NAME} PRIVATE ${XRT_LIB})

target_link_libraries(${TARGET_NAME} PRIVATE
  xrt_coreutil
  uuid
)
