#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.
# (c) Copyright 2024 Advanced Micro Devices Inc.

llvm_map_components_to_libnames(llvm_libs support)

add_llvm_tool(aie-translate aie-translate.cpp)
llvm_update_compile_flags(aie-translate)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${translation_libs}
  $<IF:$<BOOL:${CLANGIR_MLIR_FRONTEND}>,
    clangCIR
    clangCIRLoweringDirectToLLVM
    clangCIRLoweringThroughMLIR
  ,>
  AIE
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  AIETargets
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTargetAIEVecCpp
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  MLIRXLLVMToLLVMIRTranslation
)

if (CLANGIR_MLIR_FRONTEND)
  target_compile_definitions(aie-translate PRIVATE CLANGIR_MLIR_FRONTEND)
endif()

target_link_libraries(aie-translate PRIVATE ${LIBS})
