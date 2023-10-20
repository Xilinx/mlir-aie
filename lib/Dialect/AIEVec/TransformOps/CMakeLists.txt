#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023, Advanced Micro Devices, Inc.

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
