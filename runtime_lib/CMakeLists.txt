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

set(AIE_RUNTIME_TEST_TARGET_VAL ${AIR_RUNTIME_TEST_TARGET})
if(NOT x86_64_TOOLCHAIN_FILE)
  set(x86_64_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/toolchainFiles/toolchain_x86_64.cmake")
endif()
# For now, using the same toolchain file for x86_64 and HSA, can seperate if needed
if(NOT x86_64-hsa_TOOLCHAIN_FILE)
  set(x86_64-hsa_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/toolchainFiles/toolchain_x86_64.cmake")
endif()
if(NOT aarch64_TOOLCHAIN_FILE)
  if(SysrootAarch64)
  set(aarch64_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/toolchainFiles/toolchain_clang_crosscomp_pynq.cmake")
  else()
  set(aarch64_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modulesXilinx/toolchain_vitis_crosscomp_arm.cmake")
  endif()
endif()

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
  if (DEFINED LibXAIE_${target}_DIR)
    message("Using xaiengine from LibXAIE_${target}_DIR=${LibXAIE_${target}_DIR}.")
    set(LibXAIE_ROOT ${LibXAIE_${target}_DIR})
    find_package(LibXAIE REQUIRED)
    message(STATUS "FOUND LibXAIE: ${LibXAIE_FOUND}, XILINX_XAIE_LIB_DIR: ${XILINX_XAIE_LIB_DIR}, XILINX_XAIE_INCLUDE_DIR: ${XILINX_XAIE_INCLUDE_DIR}")

  elseif (DEFINED VITIS_ROOT)
  
    # if no LibXAIE_{target}_DIR specified, build xaiengine_${target} from Vitis
    message("LibXAIE_${target}_DIR not specified: Building xaiengine for ${target} from Vitis at ${VITIS_ROOT}.") 
    set(LibXAIE_FOUND TRUE)
    set(XILINX_XAIE_INCLUDE_DIR ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)
    set(XILINX_XAIE_LIBS ${VITIS_ROOT}/aietools/lib/lnx64.o/libxaiengine.so)

  endif()

  if (LibXAIE_FOUND)
    set(testLibDependsOnXaiengine xaiengine_${target}-build)
    ExternalProject_Add(xaiengine_${target}
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xaiengineTmp/${target}
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/xaiengine
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target}/xaiengine
      CMAKE_CACHE_ARGS
          -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
      CMAKE_ARGS
          -DCMAKE_TOOLCHAIN_FILE=${${target}_TOOLCHAIN_FILE}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
          -DAIE_RUNTIME_TARGET=${target}
          -DAIERT_INCLUDE_DIR=${XILINX_XAIE_INCLUDE_DIR}
          -DAIERT_LIBS=${XILINX_XAIE_LIBS}
          -DVITIS_ROOT=${VITIS_ROOT}
          -DVITIS_AIETOOLS_DIR=${VITIS_AIETOOLS_DIR}
          -DVitis_VERSION_MAJOR=${Vitis_VERSION_MAJOR}
          -DVitis_VERSION_MINOR=${Vitis_VERSION_MINOR}
          -DSysroot=${Sysroot}
      BUILD_ALWAYS true
      STEP_TARGETS clean build install test
      USES_TERMINAL_CONFIGURE true
      USES_TERMINAL_BUILD true
      USES_TERMINAL_TEST true
      USES_TERMINAL_INSTALL true
      TEST_BEFORE_INSTALL true
      TEST_EXCLUDE_FROM_MAIN true
    )
  else()
    message(STATUS "LibXAIE_${target}_DIR not specified and Vitis not found.  LibXAIE not built.")
  endif()

  # test_lib
  if (DEFINED VITIS_ROOT OR DEFINED LibXAIE_${target}_DIR)
    message(STATUS "setting up test_lib")
    ExternalProject_Add(test_lib_${target}
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/test_libTmp/${target}
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/test_lib
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target}/test_lib/lib
      CMAKE_CACHE_ARGS
          -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
      CMAKE_ARGS
          -DCMAKE_TOOLCHAIN_FILE=${${target}_TOOLCHAIN_FILE}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
          -DAIE_RUNTIME_TARGET=${target}
          -DLibXAIE_INC_DIR=${XILINX_XAIE_INCLUDE_DIR}
          -DVITIS_ROOT=${VITIS_ROOT}
          -DVITIS_AIETOOLS_DIR=${VITIS_AIETOOLS_DIR}
          -DSysroot=${Sysroot}
          -Dhsa-runtime64_DIR=${hsa-runtime64_DIR}
          -Dhsakmt_DIR=${hsakmt_DIR}
      DEPENDS ${testLibDependsOnXaiengine}
      BUILD_ALWAYS true
      STEP_TARGETS clean build install test
      USES_TERMINAL_CONFIGURE true
      USES_TERMINAL_BUILD true
      USES_TERMINAL_TEST true
      USES_TERMINAL_INSTALL true
      TEST_BEFORE_INSTALL true
      TEST_EXCLUDE_FROM_MAIN true
    )
  endif()
endforeach()

set(BOOTGEN_SOURCE ../third_party/bootgen)
set(libsources
    cdo-alloc.c
    cdo-binary.c
    cdo-command.c
    cdo-load.c
    cdo-metadata.c
    cdo-npi.c
    cdo-overlay.c
    cdo-raw.c
    cdo-source.c
    cJSON.c
    authentication.cpp
    authentication-versal.cpp
    authentication-zynq.cpp
    authentication-zynqmp.cpp
    authkeys.cpp
    authkeys-versal.cpp
    bifoptions.cpp
    bif.tab.cpp
    bif.yy.cpp
    binary.cpp
    binary-versal.cpp
    binfile.cpp
    bitutils.cpp
    bitutils-versal.cpp
    bootheader.cpp
    bootheader-versal.cpp
    bootheader-zynq.cpp
    bootheader-zynqmp.cpp
    bootimage.cpp
    bootimage-versal.cpp
    bootimage-zynq.cpp
    bootimage-zynqmp.cpp
    checksum.cpp
    checksum-versal.cpp
    cmdoptions.tab.cpp
    cmdoptions.yy.cpp
    elftools.cpp
    encryption.cpp
    encryption-versal.cpp
    encryption-versalnet.cpp
    encryption-zynq.cpp
    encryption-zynqmp.cpp
    encryptutils.cpp
    hash.cpp
    imageheadertable.cpp
    imageheadertable-versal.cpp
    imageheadertable-zynq.cpp
    imageheadertable-zynqmp.cpp
    kdf.cpp
    Keccak-compact.cpp
    Keccak-compact-versal.cpp
    logger.cpp
    # main.cpp
    mcsfile.cpp
    options.cpp
    outputfile.cpp
    outputfile-versal.cpp
    parsing.cpp
    partition.cpp
    partitionheadertable.cpp
    partitionheadertable-versal.cpp
    partitionheadertable-zynq.cpp
    partitionheadertable-zynqmp.cpp
    partition-versal.cpp
    postprocess.cpp
    readimage.cpp
    readimage-versal.cpp
    readimage-zynq.cpp
    readimage-zynqmp.cpp
    reginit.cpp
    reginit.tab.cpp
    reginit.yy.cpp
    verifyimage-versal.cpp
    verifyimage-zynqmp.cpp
)
list(TRANSFORM libsources PREPEND ${BOOTGEN_SOURCE}/)

if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows|Darwin")
# Windows fails CI because OpenSSL is not installed
# Macos fails because no malloc.h???
add_library(Bootgen STATIC ${libsources})
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(bootgen_warning_ignores
    -Wno-reorder
    -Wno-unused-private-field
    -Wno-deprecated-register
    -Wno-overloaded-virtual
    -Wno-deprecated-declarations
    -Wno-pointer-bool-conversion
    -Wno-unused-const-variable
    -Wno-tautological-overlap-compare
    -Wno-dynamic-class-memaccess
    -Wno-misleading-indentation
    -Wno-suggest-override
    -Wno-non-virtual-dtor
    -Wno-deprecated-copy
    -Wno-cast-qual
    -Wno-date-time
    -Wno-implicit-fallthrough
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-sign-compare
    -Wno-covered-switch-default
  )
else()
  set(bootgen_warning_ignores)
endif()
target_compile_options(Bootgen PRIVATE ${bootgen_warning_ignores})
target_include_directories(Bootgen PUBLIC ${BOOTGEN_SOURCE})

add_executable(bootgen ${BOOTGEN_SOURCE}/main.cpp)
target_include_directories(bootgen PUBLIC ${BOOTGEN_SOURCE})
target_compile_options(bootgen PRIVATE ${bootgen_warning_ignores})
target_link_libraries(bootgen Bootgen Bootgen ssl crypto)

install(TARGETS bootgen Bootgen)
endif() #NOT MSVC

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  include("xaiengine/aiert.cmake")
  set(XAIE_SOURCE_DIR "xaiengine/aie-rt/driver/src")

  add_aiert_headers(xaienginecdo_static
          ${XAIE_SOURCE_DIR}
          ${AIE_BINARY_DIR}/include
          ${CMAKE_INSTALL_PREFIX}/include/xaiengine
  )
  add_aiert_library(xaienginecdo_static ${XAIE_SOURCE_DIR} STATIC)

  target_link_libraries(xaienginecdo_static PUBLIC cdo_driver)
  target_compile_definitions(xaienginecdo_static PRIVATE -D__AIECDO__)

  # just to make cmake happy since AIETargets will need to re-export
  # make sure not abs path...
  # https://github.com/Kitware/CMake/blob/16b1f0c3e99023316b12fdabd1980a18b738d942/Source/cmExportInstallFileGenerator.cxx#L224
  install(TARGETS xaienginecdo_static DESTINATION lib EXPORT xaienginecdo_static)
  install(EXPORT xaienginecdo_static DESTINATION lib/cmake/aie)
  install(IMPORTED_RUNTIME_ARTIFACTS
    cdo_driver
    COMPONENT cdo_driver
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()
