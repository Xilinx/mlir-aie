#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_library(AIEVisualize
  AIEVisualize.cpp
  ArrowRenderer.cpp
  ChannelArrowRenderer.cpp

  PARTIAL_SOURCES_INTENDED
  ENABLE_AGGREGATION

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SRC_DIR}/../../include/aie/Visualize

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIETypesIncGen
  MLIRAIEXIncGen

  LINK_LIBS PUBLIC
  AIE
  AIEX
)
