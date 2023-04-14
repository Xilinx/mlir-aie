#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# optional parameters
#  -DTOOLCHAINFILES_PATH: path to cross compilation toolchain files, default: <mlir-aie-root-dir>/cmake/toolchainFiles
#  -DVitisSysroot: 
#     * when not set, x86 compile only, lit compile test for x86; --> not yet fully supported because of many hard coded --host-target=aarch64-linux-gnu in lit tests they will fail
#     * set to empty string, x86 tools compile, lit cross compile tests with default Vitis sysroot 
#     * set to valid sysroot path, x86 tools compile, lit cross compile with specified sysroot
#  -DLibXAIE_DIR: extra search path for libxaiengine.so
#  -DAIE_RUNTIME_TARGETS: list of targets (x86,aarch64) to build runtime libs for, default: x86

cmake_minimum_required(VERSION 3.21)

set(TOOLCHAINFILES_PATH ${CMAKE_SOURCE_DIR}/cmake/modulesXilinx)

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

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif() 

project(AIE LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(AIE_COMPILER "XCHESSCC" CACHE STRING "Backend compiler selection")
set(AIE_LINKER "XCHESSCC" CACHE STRING "Backend linker selection")
set(HOST_COMPILER "CLANG" CACHE STRING "Host program compiler selection")
set(PEANO_INSTALL_DIR "<unset>" CACHE STRING "Location of Peano compiler")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)
option(AIE_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the mlir-aie integration tests." OFF)
option(AIE_ENABLE_BINDINGS_PYTHON "Enable MLIR python bindings." OFF)

find_package(MLIR REQUIRED CONFIG)
find_package(Boost REQUIRED)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# These are a bit of a hack, because we're hijacking alot of LLVM machinery
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(AIE_TOOLS_BINARY_DIR ${AIE_BINARY_DIR}/bin)

find_package(Vitis COMPONENTS AIE AIE2)
configure_file(./utils/vitisVariables.config.in ${CMAKE_BINARY_DIR}/utils/vitisVariables.config @ONLY)
find_package(Python3 COMPONENTS Interpreter)

# NEEDS future improvement: if needed find default sysroot since many lit test hard code --host-target=aarch64-linux-gnu as well as tutorial defaults (in makefile-common)
if(VitisSysroot STREQUAL "" AND VITIS_ROOT)
set(VitisSysroot ${VITIS_ROOT}/gnu/aarch64/lin/aarch64-linux/aarch64-xilinx-linux/)
set(LIBCXX_VERSION "11.2.0")
set(extraAieCcFlags "-I${CMAKE_INSTALL_PREFIX}/runtime_lib/xaiengine/include -I${VitisSysroot}/usr/include -I${VitisSysroot}/usr/include/c++/${LIBCXX_VERSION} -I${VitisSysroot}/usr/include/c++/${LIBCXX_VERSION}/aarch64-xilinx-linux -L${VitisSysroot}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION} -L${CMAKE_INSTALL_PREFIX}/runtime_lib/xaiengine/lib -B${VitisSysroot}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION}")
message (STATUS "Empty string specified as sysroot, using default Vitis Sysroot to enable more unit tests: ${VitisSysroot}")
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
include(ExternalProject)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Make sure we generate the headers
add_custom_target(aie-headers)
set_target_properties(aie-headers PROPERTIES FOLDER "Misc")
add_dependencies(aie-headers mlir-headers)
# Make sure we build the docs
add_custom_target(docs ALL)
add_subdirectory(docs)
add_dependencies(docs mlir-doc)

# python install directory
if (AIE_ENABLE_BINDINGS_PYTHON)

  if(NOT AIE_PYTHON_PACKAGES_DIR)
    set(AIE_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
  endif()

  # python install directory
  if(NOT AIE_PYTHON_INSTALL_DIR)
    set(AIE_PYTHON_INSTALL_DIR "python")
  endif()

  # setup python
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  include(MLIRDetectPythonEnv)
  mlir_detect_pybind11_install()
  find_package(pybind11 2.6 REQUIRED)

endif()

add_subdirectory(include)
add_subdirectory(lib)

add_custom_target(runtime-libs ALL)

# FIXME: don't ship as source... compile them instead.
set(INSTALLS
    test_library.cpp
    test_library.h)

foreach(file ${INSTALLS})
    add_custom_target(aie-copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/runtime_lib/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/runtime_lib/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/runtime_lib/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/runtime_lib/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime_lib/${file})
    # add_dependencies(runtime-libs aie-copy-runtime-libs-${file} )
endforeach()

# Install too
list(TRANSFORM INSTALLS PREPEND "runtime_lib/")
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)

set(AIE_RUNTIME_TARGETS "x86" CACHE STRING "Architectures to compile the runtime libraries for.")
set(AIE_RUNTIME_TEST_TARGET "" CACHE STRING "Runtime architecture to test with.")
set(AIE_RUNTIME_TEST_TARGET_VAL ${AIR_RUNTIME_TEST_TARGET})

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
  ExternalProject_Add(aie_runtime_lib_${target}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/runtime_libTmp/${target}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/runtime_lib
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/runtime_lib/${target}
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
            -DLibXAIE_ROOT=${LibXAIE_ROOT}
            -DVITIS_ROOT=${VITIS_ROOT}
            -DAIE_DIR=${AIE_DIR}
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

add_subdirectory(aie_runtime_lib)

add_subdirectory(reference_designs)
add_subdirectory(test)
add_subdirectory(tutorials)
add_subdirectory(tools)

if (AIE_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/aie
    DESTINATION include
    COMPONENT aie-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${PROJECT_BINARY_DIR}/include/aie
    DESTINATION include
    COMPONENT aie-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-aie-headers
                             DEPENDS aie-headers
                             COMPONENT aie-headers)
  endif()
endif()

add_subdirectory(cmake/modules)
