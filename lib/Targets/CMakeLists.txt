#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(AIEVecToCpp)

# for #include <elf.h>
set(BOOTGEN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/bootgen)

add_mlir_library(AIERT
  AIERT.cpp
  AIETargetShared.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SRC_DIR}/../../../include/aie/Targets
)

target_link_libraries(AIERT PRIVATE xaienginecdo_static)
target_include_directories(AIERT SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
target_include_directories(obj.AIERT SYSTEM PRIVATE
  ${BOOTGEN_SOURCE_DIR}
  ${AIE_BINARY_DIR}/include/xaienginecdo_static
)
add_dependencies(obj.AIERT xaienginecdo_static xaienginecdo_static-headers)

add_mlir_library(AIETargets
  AIETargets.cpp
  AIETargetBCF.cpp
  AIETargetCDODirect.cpp
  AIETargetNPU.cpp
  AIETargetLdScript.cpp
  AIETargetXAIEV2.cpp
  AIETargetHSA.cpp
  AIETargetShared.cpp
  AIETargetSimulationFiles.cpp
  ADFGenerateCppGraph.cpp
  AIEFlowsToJSON.cpp
  AIELLVMLink.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SRC_DIR}/../../../include/aie/Targets

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
  AIERT
  AIE
  AIEX
  AIEXUtils
  ADF
)

if(AIE_ENABLE_AIRBIN)
  add_mlir_library(AIETargetAirbin
    AIETargetAirbin.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_COMPONENTS
    Support

    LINK_LIBS PRIVATE
    elf

    LINK_LIBS PUBLIC
    AIE
    AIEX
  )
  target_link_libraries(AIETargets PUBLIC AIETargetAirbin)
  target_compile_definitions(obj.AIETargets PRIVATE AIE_ENABLE_AIRBIN)
endif()

target_link_libraries(AIETargets PRIVATE xaienginecdo_static)
target_include_directories(AIETargets SYSTEM PRIVATE
  ${BOOTGEN_SOURCE_DIR}
  ${AIE_BINARY_DIR}/include/xaienginecdo_static
)
target_include_directories(obj.AIETargets SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
