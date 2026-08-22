#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Object-file analyses that are independent of MLIR: they read compiled
# artifacts, not IR, so they are linkable (and unit-testable) on their own
# rather than living inside a driver.
add_mlir_library(AIEAnalysis
  StackSizeAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aie/Analysis

  LINK_COMPONENTS
  Object
  Support
)
