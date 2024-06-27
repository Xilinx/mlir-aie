#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Xilinx Inc.

add_executable(aie-visualize aie-visualize.cpp)

target_include_directories(aie-visualize PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_update_compile_flags(aie-visualize)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(aie-visualize ${llvm_libs})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# configure_file(configure.h.in configure.h)
# target_include_directories(aie-visualize PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(aie-visualize
  ${dialect_libs}
  MLIRParser
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRAIEVecDialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRXLLVMToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  ADF
  AIE
  AIETransforms
  AIETargets
  AIEX
  AIEXTransforms
  MLIRAIEVecDialect
  MLIRXLLVMDialect)

install(TARGETS aie-visualize
  EXPORT AIE-VISUALIZE
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aie-visualize)
