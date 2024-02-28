# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_public_c_api_library(AIECAPI
  Dialects.cpp
  Registration.cpp
  Translation.cpp

  LINK_LIBS PUBLIC
  ADF
  AIE
  AIETargets
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEVecDialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRAIEVecUtils
  MLIRTargetAIEVecCpp
)

if(AIE_ENABLE_GENERATE_CDO_DIRECT)
  target_include_directories(AIECAPI PUBLIC ${VITIS_AIETOOLS_DIR}/include)
  target_compile_definitions(obj.AIECAPI PRIVATE AIE_ENABLE_GENERATE_CDO_DIRECT)
endif()

