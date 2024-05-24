#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(_aie2xclbin_srcs aie2xclbin.cpp XCLBinGen.cpp)

add_executable(aie2xclbin ${_aie2xclbin_srcs})

target_include_directories(aie2xclbin PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_update_compile_flags(aie2xclbin)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(aie2xclbin ${llvm_libs})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

configure_file(configure.h.in configure.h)
target_include_directories(aie2xclbin PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(aie2xclbin
  ${dialect_libs}
  MLIRParser
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  MLIRAIEVecDialect
  MLIRAIEVecToLLVM
  MLIRAIEVecTransforms
  MLIRXLLVMToLLVMIRTranslation
  ADF
  AIE
  AIETransforms
  AIETargets
  AIEX
  AIEXTransforms
  MLIRAIEVecDialect
  MLIRXLLVMDialect
  ${UUID})

install(TARGETS aie2xclbin
  EXPORT AIE2XCLBIN
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aie2xclbin)
