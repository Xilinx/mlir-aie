#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(AIEVecToCpp)

set(LLVM_OPTIONAL_SOURCES AIETargetCDODirect.cpp AIETargetAirbin.cpp)

set(_sources
  AIETargets.cpp
  AIETargetBCF.cpp
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

if(AIE_ENABLE_AIRBIN)
  list(APPEND _sources AIETargetAirbin.cpp)
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
  target_link_libraries(AIETargets PUBLIC xaienginecdo_static)
  target_include_directories(AIETargets PUBLIC ${VITIS_AIETOOLS_DIR}/include)
  add_dependencies(obj.AIETargets xaienginecdo_static xaienginecdo_static-headers)
  target_compile_definitions(obj.AIETargets PRIVATE AIE_ENABLE_GENERATE_CDO_DIRECT)
endif()

if(AIE_ENABLE_AIRBIN)
  target_link_libraries(AIETargets PRIVATE elf)
  target_compile_definitions(obj.AIETargets PRIVATE AIE_ENABLE_AIRBIN)
endif()
