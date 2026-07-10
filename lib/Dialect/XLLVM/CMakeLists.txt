#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRXLLVMDialect
  XLLVMOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/XLLVM

  DEPENDS
  MLIRXLLVMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )
