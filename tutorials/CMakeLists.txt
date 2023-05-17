#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(aie-test LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(ENABLE_CHESS_TESTS ON CACHE BOOL "Enable backend tests using xchesscc")
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
  set(DEFAULT_ENABLE_BOARD_TESTS ON)
else()
  set(DEFAULT_ENABLE_BOARD_TESTS OFF)
endif()
option(ENABLE_BOARD_TESTS "Enable board tests" ${DEFAULT_ENABLE_BOARD_TESTS})

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

find_package(Vitis 2022.2 COMPONENTS AIE AIE2)
find_package(Python3 COMPONENTS Interpreter)

# Look for LibXAIE
get_directory_property(LibXAIE_ROOT
  DIRECTORY ${CMAKE_SOURCE_DIR}/runtime_lib
  DEFINITION LibXAIE_ROOT)
get_directory_property(XILINX_XAIE_INCLUDE_DIR
  DIRECTORY ${CMAKE_SOURCE_DIR}/runtime_lib
  DEFINITION XILINX_XAIE_INCLUDE_DIR)
if(DEFINED LibXAIE_ROOT)
  message(STATUS "Libxaie will be compiled and picked up from ${LibXAIE_ROOT}")
  set(LibXAIE_FOUND TRUE)
else()
  if(DEFINED VITIS_ROOT)
    message(STATUS "Vitis available, libxaie will be compiled and picked up locally")
    set(LibXAIE_FOUND TRUE)
  endif()
endif()

if(NOT AIE_BINARY_DIR)
find_package(AIE REQUIRED CONFIG)
endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

if(ENABLE_CHESS_TESTS)
set(CONFIG_ENABLE_CHESS_TESTS 1)
else()
set(CONFIG_ENABLE_CHESS_TESTS 0)
endif()
if(ENABLE_BOARD_TESTS)
set(CONFIG_ENABLE_BOARD_TESTS 1)
else()
set(CONFIG_ENABLE_BOARD_TESTS 0)
endif()
if(LibXAIE_FOUND)
set(CONFIG_HAS_LIBXAIE 1)
else()
set(CONFIG_HAS_LIBXAIE 0)
endif()
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  aiecc.py
  aie-opt
  aie-translate
  )

add_lit_testsuite(check-tutorials "Running the aie tutorials tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS "-sv --timeout 300 --time-tests"
  )
set_target_properties(check-tutorials PROPERTIES FOLDER "Tutorials")

add_lit_testsuites(TUTORIALS ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TEST_DEPENDS} ARGS "-sv --timeout 300 --time-tests")
