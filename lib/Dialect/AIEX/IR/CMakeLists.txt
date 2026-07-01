#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(AIEX
  AIEXDialect.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEXIncGen
  MLIRAIEXEnumsIncGen

  LINK_LIBS PUBLIC
  AIE
  MLIRIR
  MLIRSupport
  )
