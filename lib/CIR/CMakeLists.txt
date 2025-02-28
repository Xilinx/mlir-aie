#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

add_mlir_dialect_library(
  CIRToAIEPasses
  CIRToAIEPasses.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRCIRToAIEPassesIncGen

  LINK_LIBS PUBLIC
  clangCIR
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect)
