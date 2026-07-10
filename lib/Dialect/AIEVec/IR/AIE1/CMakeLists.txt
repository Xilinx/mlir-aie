#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecAIE1Dialect
  AIEVecAIE1Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/aie/Dialect/AIEVec/

  DEPENDS
  MLIRAIEVecAIE1OpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )
