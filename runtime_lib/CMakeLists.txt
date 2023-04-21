#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# cmake needs this line
cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

project("runtime-libs")
#add_custom_target(runtime-libs ALL)

set(AIE_RUNTIME_TARGETS "x86_64" CACHE STRING "Architectures to compile the runtime libraries for.")
set(AIE_RUNTIME_TEST_TARGET "x86_64" CACHE STRING "Runtime architecture to test with.")
set(AIE_RUNTIME_TEST_TARGET_VAL ${AIR_RUNTIME_TEST_TARGET})
set(x86_64_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/toolchainFiles/toolchain_x86_64.cmake")
set(aarch64_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modulesXilinx/toolchain_vitis_crosscomp_arm.cmake")

if (DEFINED VITIS_ROOT)
foreach(target ${AIE_RUNTIME_TARGETS})
  # By default, we test the first architecture in AIE_RUNTIME_TARGETS.
  # Alternatively, this can be defined to force testing with a particular architecture.
  if (NOT AIE_RUNTIME_TEST_TARGET_VAL)
    set(AIE_RUNTIME_TEST_TARGET_VAL ${target})
    message("Testing with AIE runtime target: ${AIE_RUNTIME_TEST_TARGET_VAL}")
  endif()

  if (NOT EXISTS ${${target}_TOOLCHAIN_FILE})
    message(FATAL_ERROR "Toolchain file ${${target}_TOOLCHAIN_FILE} not found! Cannot build target ${target}.")
  endif()
  message("Building AIE runtime for ${target} with TOOLCHAIN=${${target}_TOOLCHAIN_FILE}")
  
  # xaiengine
  ExternalProject_Add(xaiengine_${target}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xaiengineTmp/${target}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/xaiengine
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target}/xaiengine
    CMAKE_CACHE_ARGS
            -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
    CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${${target}_TOOLCHAIN_FILE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
            -DAIE_RUNTIME_TARGET=${target}
            -DVITIS_ROOT=${VITIS_ROOT}
            -DVITIS_AIETOOLS_DIR=${VITIS_AIETOOLS_DIR}
    BUILD_ALWAYS true
    STEP_TARGETS clean build install test
    USES_TERMINAL_CONFIGURE true
    USES_TERMINAL_BUILD true
    USES_TERMINAL_TEST true
    USES_TERMINAL_INSTALL true
    TEST_BEFORE_INSTALL true
    TEST_EXCLUDE_FROM_MAIN true
  )

  # test_lib
  ExternalProject_Add(test_lib_${target}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/test_libTmp/${target}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/test_lib
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target}/test_lib/lib
  CMAKE_CACHE_ARGS
          -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
  CMAKE_ARGS
          -DCMAKE_TOOLCHAIN_FILE=${${target}_TOOLCHAIN_FILE}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
          -DAIE_RUNTIME_TARGET=${target}
  #DEPENDS xaiengine_${target}
  BUILD_ALWAYS true
  STEP_TARGETS clean build install test
  USES_TERMINAL_CONFIGURE true
  USES_TERMINAL_BUILD true
  USES_TERMINAL_TEST true
  USES_TERMINAL_INSTALL true
  TEST_BEFORE_INSTALL true
  TEST_EXCLUDE_FROM_MAIN true
)
endforeach()
endif()
