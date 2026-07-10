#
# Copyright (C) 2021-2022 Xilinx, Inc.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

llvm_map_components_to_libnames(llvm_libs support)

add_llvm_tool(aie-translate aie-translate.cpp)
llvm_update_compile_flags(aie-translate)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${translation_libs}
  AIE
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  AIETargets
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRXLLVMToLLVMIRTranslation
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)

target_link_libraries(aie-translate PRIVATE ${LIBS})
