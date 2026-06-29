# Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_translation_library(MLIRXLLVMToLLVMIRTranslation
  XLLVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRXLLVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRXLLVMDialect
  )

