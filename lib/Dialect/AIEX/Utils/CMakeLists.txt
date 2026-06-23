#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.

add_mlir_dialect_library(AIEXUtils
  AIETokenAnalysis.cpp
  AIEUtils.cpp

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
  )
