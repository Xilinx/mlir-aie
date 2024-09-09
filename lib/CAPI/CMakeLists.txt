# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_public_c_api_library(AIECAPI
  Dialects.cpp
  Registration.cpp
  TargetModel.cpp
  Translation.cpp

  LINK_LIBS PUBLIC
  ADF
  AIE
  AIETargets
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEToTransaction
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRAIEVecUtils
  MLIRTargetAIEVecCpp
  MLIRXLLVMToLLVMIRTranslation
)

target_include_directories(AIECAPI PUBLIC ${VITIS_AIETOOLS_DIR}/include)

