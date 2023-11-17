#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_subdirectory(AIEVecToCpp)

add_mlir_library(AIETargets
  AIETargets.cpp
  AIETargetCDO.cpp
  AIETargetIPU.cpp
  AIETargetXAIEV2.cpp
  AIETargetShared.cpp
  AIETargetSimulationFiles.cpp
  ADFGenerateCppGraph.cpp
  AIEFlowsToJSON.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  LINK_COMPONENTS
  Core
  IRReader
  Support
  TransformUtils

  LINK_LIBS PUBLIC
  AIE
  AIEX
  AIEXUtils
  ADF
)
