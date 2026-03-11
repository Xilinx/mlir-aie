# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
add_mlir_conversion_library(MLIRAIEXToEmitC
  AIEXToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  $(CMAKE_CURRENT_SRC_DIR)/../../../../include/aie/Conversion/AIEXToEmitC

  DEPENDS
  MLIRAIEConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIE
  AIEX
  MLIREmitCDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransforms
)
