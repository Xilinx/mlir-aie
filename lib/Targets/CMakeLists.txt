#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(AIEVecToCpp)

set(LLVM_OPTIONAL_SOURCES AIETargetCDODirect.cpp)

set(_sources
  AIETargets.cpp
  AIETargetBCF.cpp
  AIETargetCDO.cpp
  AIETargetIPU.cpp
  AIETargetLdScript.cpp
  AIETargetXAIEV2.cpp
  AIETargetShared.cpp
  AIETargetSimulationFiles.cpp
  ADFGenerateCppGraph.cpp
  AIEFlowsToJSON.cpp
  AIELLVMLink.cpp
)

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  list(APPEND _sources AIETargetCDODirect.cpp)
endif()

add_mlir_library(AIETargets
  ${_sources}

  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include
  $(CMAKE_CURRENT_SRC_DIR)/../../../include/aie/Targets

  LINK_COMPONENTS
  BinaryFormat
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Support
  TransformUtils
  IPO

  LINK_LIBS PUBLIC
  AIE
  AIEX
  AIEXUtils
  ADF
)

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  message(STATUS "Building CDO direct")
  include("${AIE_SOURCE_DIR}/runtime_lib/xaiengine/aiert.cmake")
  set(XAIE_SOURCE_DIR "${AIE_SOURCE_DIR}/runtime_lib/xaiengine/aie-rt/driver/src")

  add_aiert_headers(xaienginecdo
    ${XAIE_SOURCE_DIR}
    ${AIE_BINARY_DIR}/include
    ${CMAKE_INSTALL_PREFIX}/include/xaiengine
  )
  add_aiert_library(xaienginecdo ${XAIE_SOURCE_DIR} STATIC)

  # TODO: set RPATH correctly (and for XRTModule)
  add_library(cdo_driver SHARED IMPORTED)
  set_property(TARGET cdo_driver PROPERTY IMPORTED_LOCATION "${CDO_DRIVER_LIBRARY}")

  target_link_libraries(xaienginecdo PUBLIC cdo_driver)
  target_compile_definitions(xaienginecdo PRIVATE -D__AIECDO__)

  target_link_libraries(AIETargets PUBLIC xaienginecdo)
  add_dependencies(AIETargets xaienginecdo xaienginecdo-headers)
  target_compile_definitions(obj.AIETargets PRIVATE AIE_ENABLE_GENERATE_CDO_DIRECT)

  # just to make cmake happy since AIETargets will need to re-export
  install(TARGETS xaienginecdo DESTINATION ${CMAKE_INSTALL_PREFIX}/lib EXPORT xaienginecdo)
  install(EXPORT xaienginecdo DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)
  install(IMPORTED_RUNTIME_ARTIFACTS
    cdo_driver
    COMPONENT aie-translate-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
endif()