# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
  MLIRAIEToConfiguration
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRAIEVecUtils
  MLIRXLLVMToLLVMIRTranslation
)

target_include_directories(AIECAPI PUBLIC ${VITIS_AIETOOLS_DIR}/include)

