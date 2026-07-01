# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)
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

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0116 OLD)
endif()

project(PI LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

if(MSVC)
  add_compile_options(
    $<$<CONFIG:>:/MT> # ---------|
    $<$<CONFIG:Debug>:/MTd> # ---|-- Statically link the runtime libraries
    $<$<CONFIG:Release>:/MT> # --|
  )
endif()

find_package(AIE REQUIRED CONFIG)

message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${AIE_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})

include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aie.")

set(_sources
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  MLIRPythonExtension.ExecutionEngine
  AIEPythonExtensions
  AIEPythonSources
  AIEPythonSources.Compiler
)

add_mlir_python_common_capi_library(AIEAggregateCAPI
  INSTALL_COMPONENT AIEPythonModules
  INSTALL_DESTINATION aie/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/aie/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_sources}
)

# ##############################################################################
# Custom targets.
# ##############################################################################

_flatten_mlir_python_targets(aie_python_sources_deps
  AIEPythonSources
  AIEPythonSources.Compiler
)
add_custom_target("aie-python-sources" DEPENDS ${aie_python_sources_deps})
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-aie-python-sources DEPENDS
                           aie-python-sources COMPONENT aie-python-sources)
endif()

file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/aie/compiler/aiecc"
  TYPE FILE
  FILES "${AIE_PYTHON_INSTALL_DIR}/aie/compiler/aiecc/configure.py"
)

# ##############################################################################
# The fully assembled package of modules. This must come last.
# ##############################################################################

add_mlir_python_modules(AIEMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/aie"
  INSTALL_PREFIX aie
  DECLARED_SOURCES ${_sources}
  COMMON_CAPI_LINK_LIBS
    AIEAggregateCAPI
)
