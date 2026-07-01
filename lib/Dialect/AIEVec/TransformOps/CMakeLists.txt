#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecTransformOps
  DialectExtension.cpp
  AIEVecTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec/TransformOps

  DEPENDS
  MLIRAIEVecTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransformDialect
  MLIRVectorDialect
  )
