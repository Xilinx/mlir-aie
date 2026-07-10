#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# for #include <elf.h>
set(BOOTGEN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/bootgen)

add_mlir_library(AIERT
  AIERT.cpp
  AIETargetShared.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SRC_DIR}/../../../include/aie/Targets

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEXIncGen
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
  AIETargetCppTxn.cpp
  AIETargetNPU.cpp
  AIETargetLdScript.cpp
  AIETargetUcCert.cpp
  AIETargetXAIEV2.cpp
  AIETargetHSA.cpp
  AIETargetShared.cpp
  AIETargetSimulationFiles.cpp
  AIEFlowsToJSON.cpp
  AIELLVMLink.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SRC_DIR}/../../../include/aie/Targets

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEXIncGen

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
  MLIRAIEXToEmitC
  MLIREmitCDialect
  MLIRTargetCpp
)

target_link_libraries(AIETargets PRIVATE xaienginecdo_static)
target_include_directories(AIETargets SYSTEM PRIVATE
  ${BOOTGEN_SOURCE_DIR}
  ${AIE_BINARY_DIR}/include/xaienginecdo_static
)
target_include_directories(obj.AIETargets SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR})
