#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect_library(MLIRAIEVecDialect
  AIEVecOps.cpp
  AIEVecTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/aie/Dialect/AIEVec

  DEPENDS
  MLIRAIEVecOpsIncGen
  MLIRAIEVecAttributesIncGen
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen

  LINK_LIBS PUBLIC
  AIE
  MLIRIR
  MLIRPass
  )

  add_subdirectory(AIE1)
