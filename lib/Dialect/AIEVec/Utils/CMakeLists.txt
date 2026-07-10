#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec

  LINK_LIBS PUBLIC
  MLIRAIEVecDialect
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRVectorDialect
  )
