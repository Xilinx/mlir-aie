#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

add_mlir_conversion_library(AIEXToEmitC
  AIEXToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Conversion

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  AIE
  AIEX
  MLIRIR
  MLIRPass
  MLIRTransforms
)
