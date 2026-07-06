#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(AIEXUtils
  AIETokenAnalysis.cpp
  AIEUtils.cpp
  BdLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEPassIncGen
  MLIRAIEXIncGen
  MLIRAIEXPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  )
