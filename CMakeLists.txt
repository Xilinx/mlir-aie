#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# optional parameters
#  -DTOOLCHAINFILES_PATH: path to cross compilation toolchain files, default: <mlir-aie-root-dir>/cmake/toolchainFiles
#  -DLibXAIE_x86_64_DIR: search path for x86_64 libxaiengine
#  -DLibXAIE_aarch64_DIR: search path for aarch64 libxaiengine
#  -DSysrootAarch64: sysroot for aarch crosscompile, if not specified and aarch64 in AIE_RUNTIME_TARGETS then default to  Vitis aarch64 sysroot
#  -DAIE_RUNTIME_TARGETS: list of targets (x86_64,aarch64) to build runtime libs for, default: x86_64; cross compilation for aarch64 against default Vitis Sysroot
#  -DAIE_RUNTIME_TEST_TARGET: runtime test target (x86_64 or aarch64) used for running unit test and tutorials, default x86_64

cmake_minimum_required(VERSION 3.30)
set(TOOLCHAINFILES_PATH ${CMAKE_SOURCE_DIR}/cmake/modulesXilinx)

set(AIE_RUNTIME_TARGETS
    "x86_64"
    CACHE STRING "Architectures to compile the runtime libraries for.")
set(AIE_RUNTIME_TEST_TARGET
    "x86_64"
    CACHE STRING "Runtime architecture to test with.")

set(AIE_VITIS_COMPONENTS "AIE;AIE2;AIE2P" CACHE STRING "Vitis components")

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
  cmake_policy(SET CMP0116 OLD)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(AIE LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)


set(AIE_COMPILER
    "XCHESSCC"
    CACHE STRING "Backend compiler selection")
set(AIE_LINKER
    "XCHESSCC"
    CACHE STRING "Backend linker selection")
set(HOST_COMPILER
    "NONE"
    CACHE STRING "Host program compiler selection")
set(PEANO_INSTALL_DIR
    "$ENV{PEANO_INSTALL_DIR}"
    CACHE STRING "Location of Peano compiler")
if(NOT PEANO_INSTALL_DIR OR PEANO_INSTALL_DIR STREQUAL "")
    set(PEANO_INSTALL_DIR "<unset>" CACHE STRING "Location of Peano compiler" FORCE)
endif()

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS
       "Build the LLVM tools. If OFF, just generate build targets." ON)
option(AIE_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the mlir-aie integration tests." OFF)

# Take note of the current git revision for outputting version strings in built executables
execute_process(COMMAND git rev-parse HEAD OUTPUT_VARIABLE AIE_GIT_COMMIT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# These are a bit of a hack, because we're hijacking alot of LLVM machinery
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(AIE_TOOLS_BINARY_DIR ${AIE_BINARY_DIR}/bin)
set(AIE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Vitis 2023.2 COMPONENTS ${AIE_VITIS_COMPONENTS})
configure_file(./utils/vitisVariables.config.in
               ${CMAKE_BINARY_DIR}/utils/vitisVariables.config @ONLY)
find_package(XRT)
find_package(hsa-runtime64)

# Set up default Vitis Sysroot as sysroot when testing on aarch64
list(FIND AIE_RUNTIME_TARGETS "aarch64" indexAarch64)
if(NOT DEFINED SysrootAarch64
   AND DEFINED VITIS_ROOT
   AND ${indexAarch64} GREATER -1)
  set(Sysroot ${VITIS_ROOT}/gnu/aarch64/lin/aarch64-linux/aarch64-xilinx-linux)
  if(AIE_RUNTIME_TEST_TARGET STREQUAL "aarch64")
    set(LIBCXX_VERSION "11.2.0")
    set(extraAieCcFlags
        "-I${CMAKE_INSTALL_PREFIX}/runtime_lib/aarch64/xaiengine/include -I${Sysroot}/usr/include -I${Sysroot}/usr/include/c++/${LIBCXX_VERSION} -I${Sysroot}/usr/include/c++/${LIBCXX_VERSION}/aarch64-xilinx-linux -L${Sysroot}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION} -L${CMAKE_INSTALL_PREFIX}/runtime_lib/aarch64/xaiengine/lib -B${Sysroot}/usr/lib/aarch64-xilinx-linux/${LIBCXX_VERSION}"
    )
  endif()
  message(
    STATUS
      "aarch64 specified as runtime target but no sysroot set, using default Vitis Sysroot for cross compilation: ${Sysroot}"
  )
elseif(SysrootAarch64 AND ${indexAarch64} GREATER -1)
  set(Sysroot ${SysrootAarch64})
endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-j 8 -sv")
if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS
    "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(ExternalProject)
include(CMakeDependentOption)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

cmake_dependent_option(AIE_ENABLE_BINDINGS_PYTHON
  "Enables building of Python bindings." ON "MLIR_ENABLE_BINDINGS_PYTHON" OFF)

cmake_dependent_option(AIE_ENABLE_PYTHON_PASSES
  "Enables building of passes that connect to python." OFF "AIE_ENABLE_BINDINGS_PYTHON;LLVM_ENABLE_RTTI" OFF)

find_library(XRT_COREUTIL xrt_coreutil HINTS ${XRT_LIB_DIR})
find_library(UUID uuid)
cmake_dependent_option(AIE_ENABLE_XRT_PYTHON_BINDINGS
  "Enables building of python bindings to XRT runtime." ON "XRT_COREUTIL;UUID" OFF)
if (AIE_ENABLE_XRT_PYTHON_BINDINGS)
  add_library(xrt_coreutil SHARED IMPORTED)
  set_property(TARGET xrt_coreutil PROPERTY IMPORTED_LOCATION "${XRT_COREUTIL}")
endif()

cmake_dependent_option(AIECC_COMPILE
  "Set aiecc to compile." ON "NOT AIE_COMPILER STREQUAL NONE" OFF)

cmake_dependent_option(AIECC_LINK
  "Set aiecc to link." ON "NOT AIE_LINKER STREQUAL NONE" OFF)

cmake_dependent_option(AIECC_HOST_COMPILE
  "Set aiecc to host compile." ON "NOT HOST_COMPILER STREQUAL NONE" OFF)

cmake_dependent_option(AIECC_COMPILE_WITH_XCHESSCC
  "Set aiecc to compile with chess." ON "AIECC_COMPILE;AIE_COMPILER STREQUAL XCHESSCC" OFF)

cmake_dependent_option(AIECC_LINK_WITH_XCHESSCC
  "Set aiecc to link with chess." ON "AIECC_LINK;AIE_LINKER STREQUAL XCHESSCC" OFF)

find_library(ELF_LIB elf)
cmake_dependent_option(AIE_ENABLE_AIRBIN
  "Enables emitting AIRBIN ELF binaries." OFF "ELF_LIB" OFF)

# If we need runtime libs, then statically link them.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_flag_if_supported("-Werror=sign-compare" WERROR_SIGN_COMPARE)
add_flag_if_supported("-Werror=unused" WERROR_USED)

# What happens when you have a non-void function with no return?
# No `ret` instruction is generated and so execution of that function just
# proceeds as if it doesn't have a care in the world (right into whatever comes next).
# https://godbolt.org/z/Wr9nzv6ns
add_flag_if_supported("-Werror=return-type" WERROR_RETURN_TYPE)

## Stolen from MLIR

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
add_flag_if_supported("-Werror=implicit-function-declaration" WERROR_IMPLICIT_FUNCTION_DECLARATION)

# Forbid mismatch between declaration and definition for class vs struct. This is
# harmless on Unix systems, but it'll be a ticking bomb for MSVC/Windows systems
# where it creeps into the ABI.
add_flag_if_supported("-Werror=mismatched-tags" WERROR_MISMATCHED_TAGS)

# Silence a false positive GCC -Wunused-but-set-parameter warning in constexpr
# cases, by marking SelectedCase as used. See
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827 for details. The issue is
# fixed in GCC 10.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
  add_flag_if_supported("-Wno-unused-but-set-parameter" WNO_UNUSED_BUT_SET_PARAMETER)
endif()

# Make sure we generate the headers
add_custom_target(aie-headers)
set_target_properties(aie-headers PROPERTIES FOLDER "Misc")
add_dependencies(aie-headers mlir-headers)

option(AIE_INCLUDE_DOCS "Generate build targets for the MLIR AIE docs." OFF)
if(AIE_INCLUDE_DOCS)
  # Make sure we build the docs
  add_custom_target(docs ALL)
  add_subdirectory(docs)
  add_dependencies(docs mlir-doc)
endif()

append_if(BUILD_INSTRUMENTED_COVERAGE "-O0 -fprofile-arcs -ftest-coverage -fprofile-instr-generate=code-%p-%100000000m.profraw -fcoverage-mapping"
  CMAKE_CXX_FLAGS
  CMAKE_C_FLAGS
  CMAKE_EXE_LINKER_FLAGS
  CMAKE_SHARED_LINKER_FLAGS)

if(AIE_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  if(NOT AIE_PYTHON_INSTALL_DIR)
    set(AIE_PYTHON_INSTALL_DIR "python")
  endif()

  if(NOT AIE_PYTHON_PACKAGES_DIR)
    set(AIE_PYTHON_PACKAGES_DIR "${AIE_BINARY_DIR}/${AIE_PYTHON_INSTALL_DIR}")
  endif()
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(aie_runtime_lib)
add_subdirectory(tools)
add_subdirectory(runtime_lib)

if(AIE_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY include/aie include/aie-c
    DESTINATION include
    COMPONENT aie-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT")

  install(
    DIRECTORY ${PROJECT_BINARY_DIR}/include/aie
    DESTINATION include
    COMPONENT aie-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE)

  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-aie-headers DEPENDS aie-headers COMPONENT
                             aie-headers)
  endif()
endif()


# Last because each of these has its own CMakeLists.txt which reloads/re-finds LLVM, thus resettings globals.
add_subdirectory(programming_examples)
add_subdirectory(programming_guide)
add_subdirectory(test)
add_subdirectory(mlir_exercises)
add_subdirectory(cmake/modules)

get_filename_component(COMPILER_DIRECTORY ${CMAKE_CXX_COMPILER} DIRECTORY)
find_program(LLVM_COV "llvm-cov" ${COMPILER_DIRECTORY} NO_DEFAULT_PATH)
find_program(LLVM_PROFDATA "llvm-profdata" ${COMPILER_DIRECTORY} NO_DEFAULT_PATH)
set(LLVM_PROFILE_DATA_DIR ${PROJECT_BINARY_DIR})
file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/report" REPORT_DIR)

if (NOT INSTRUMENTED_COVERAGE_FILES)
  set(INSTRUMENTED_COVERAGE_FILES
    "${AIE_SOURCE_DIR}/lib"
    "${AIE_SOURCE_DIR}/include"
    "${AIE_SOURCE_DIR}/tools"
    "${AIE_SOURCE_DIR}/python"
  )
endif()

add_custom_target(generate-aie-coverage-report
  COMMAND ${Python3_EXECUTABLE} ${AIE_SOURCE_DIR}/utils/prepare-code-coverage-artifact.py
  ${LLVM_PROFDATA} ${LLVM_COV} ${LLVM_PROFILE_DATA_DIR}
  ${REPORT_DIR}
  ${CMAKE_BINARY_DIR}/bin/aie-opt
  --unified-report --restrict ${INSTRUMENTED_COVERAGE_FILES}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS check-aie) # Run tests
