#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecTransforms
  ConvertVectorToAIEVec.cpp
  VectorToVectorConversions.cpp
  VectorToAIEVecConversions.cpp
  AIEVecOptimizations.cpp
  FoldMulAddChainToConvOp.cpp
  CopyRemoval.cpp
  DynamicSizeNoImplicitBroadcast.cpp
  SplitVectorLoadUpsChains.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec/Transforms
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec/Analysis
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec/Utils

  DEPENDS
  MLIRAIEVecPassIncGen
  MLIRAIEVecAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRAIEVecUtils)
