#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023, Advanced Micro Devices, Inc.
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
