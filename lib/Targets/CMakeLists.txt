#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(AIEVecToCpp)

set(LLVM_OPTIONAL_SOURCES AIETargetAirbin.cpp)

# for #include <elf.h>
set(BOOTGEN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/bootgen)

add_mlir_library(AIERTX
  AIERTX.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  $(CMAKE_CURRENT_SRC_DIR)/../../../include/aie/Targets
)

target_link_libraries(AIERTX PRIVATE xaienginecdo_static)
target_include_directories(AIERTX SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
target_include_directories(obj.AIERTX SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
add_dependencies(obj.AIERTX xaienginecdo_static xaienginecdo_static-headers)

set(_sources
  AIETargets.cpp
  AIETargetBCF.cpp
  AIETargetCDODirect.cpp
  AIETargetIPU.cpp
  AIETargetLdScript.cpp
  AIETargetXAIEV2.cpp
  AIETargetShared.cpp
  AIETargetSimulationFiles.cpp
  ADFGenerateCppGraph.cpp
  AIEFlowsToJSON.cpp
  AIELLVMLink.cpp
)

if(AIE_ENABLE_AIRBIN)
  list(APPEND _sources AIETargetAirbin.cpp)
endif()

add_mlir_library(AIETargets
  ${_sources}

  PARTIAL_SOURCES_INTENDED
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
  AIERTX
  AIE
  AIEX
  AIEXUtils
  ADF
)

target_link_libraries(AIETargets PRIVATE xaienginecdo_static)
target_include_directories(AIETargets SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
target_include_directories(obj.AIETargets SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})

if(AIE_ENABLE_AIRBIN)
  target_link_libraries(AIETargets PRIVATE elf)
  target_compile_definitions(obj.AIETargets PRIVATE AIE_ENABLE_AIRBIN)
endif()
