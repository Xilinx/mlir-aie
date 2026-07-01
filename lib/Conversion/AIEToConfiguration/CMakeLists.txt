# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
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
target_include_directories(MLIRAIEToConfiguration SYSTEM PRIVATE
  ${AIE_BINARY_DIR}/include/xaienginecdo_static
)
