# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

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

include(CMakeDependentOption)

if(POLICY CMP0127)
  cmake_policy(SET CMP0127 NEW)
endif()

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(aie-test LANGUAGES CXX C)

# find package AIE if running tests from AIE installation
if(NOT AIE_BINARY_DIR)
  find_package(AIE REQUIRED CONFIG)
  set(LibXAIE_${AIE_RUNTIME_TEST_TARGET}_DIR CACHE STRING "") #pick up libxaiengine from installation folder if no other location specified
endif()

# default to x86_64 system architecture for runtime target
set(AIE_RUNTIME_TARGETS "x86_64" CACHE STRING "Architectures to compile the runtime libraries for.")
list(GET AIE_RUNTIME_TARGETS 0 firstRuntimeTarget)
set(AIE_RUNTIME_TEST_TARGET ${firstRuntimeTarget} CACHE STRING "Runtime architecture to test with.")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

find_package(Vitis 2023.2 COMPONENTS AIE AIE2)
find_package(Python3 COMPONENTS Interpreter)
find_package(XRT)
find_package(hsa-runtime64)

cmake_dependent_option(ENABLE_CHESS_TESTS "Enable backend tests using xchesscc" ON "Vitis_FOUND" OFF)
cmake_dependent_option(ENABLE_PYTHON_TESTS "Enable python tests" ON "AIE_ENABLE_BINDINGS_PYTHON" OFF)

# Look for LibXAIE
set(LibXAIE_FOUND FALSE)
if (DEFINED LibXAIE_${AIE_RUNTIME_TEST_TARGET}_DIR)
    message("Test using xaiengine from LibXAIE_${AIE_RUNTIME_TEST_TARGET}_DIR=${LibXAIE_${AIE_RUNTIME_TEST_TARGET}_DIR}")
    set(LibXAIE_ROOT ${LibXAIE_${target}_DIR})
    find_package(LibXAIE)
else()
  if(DEFINED VITIS_ROOT)
    message(STATUS "Test has Vitis available, no libxaie location specified so pick up from build area")
    set(LibXAIE_FOUND TRUE)
  endif()
endif()

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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  AIEPythonModules
  aie-lsp-server
  aie-opt
  aie-translate
)

add_lit_testsuite(check-aie "Running the aie regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS "-sv --timeout 600 -j 3"
)
set_target_properties(check-aie PROPERTIES FOLDER "Tests")
