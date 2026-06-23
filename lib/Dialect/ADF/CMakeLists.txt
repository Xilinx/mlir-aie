#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc.

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

