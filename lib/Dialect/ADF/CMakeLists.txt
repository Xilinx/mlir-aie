#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(ADF
  ADFDialect.cpp
  ADFOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/ADF

  DEPENDS
  MLIRADFIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

