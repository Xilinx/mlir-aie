# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  MLIRArithToEmitC
  MLIRSCFToEmitC
  MLIREmitCDialect
  MLIREmitCTransforms
  MLIRSCFDialect
  MLIRTransforms
  )
