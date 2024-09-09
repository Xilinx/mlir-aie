#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_executable(aie-opt aie-opt.cpp)
llvm_update_compile_flags(aie-opt)
install(TARGETS aie-opt
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  ADF
  AIE
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEToTransaction
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecTransformOps
  MLIRAIEVecTransforms
  MLIRAIEVecToLLVM
  MLIRTransformDialect
  MLIRXLLVMDialect
  )
target_link_libraries(aie-opt PUBLIC ${LIBS})
