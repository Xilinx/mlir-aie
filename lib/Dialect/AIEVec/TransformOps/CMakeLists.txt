#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates

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
