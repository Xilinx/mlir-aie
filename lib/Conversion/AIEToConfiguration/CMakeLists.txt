# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
add_mlir_conversion_library(MLIRAIEToConfiguration
  AIEToConfiguration.cpp

  ADDITIONAL_HEADER_DIRS
  $(CMAKE_CURRENT_SRC_DIR)/../../../../include/aie/Conversion/AIEToConfiguration

  DEPENDS
  MLIRAIEConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIERT
  )
