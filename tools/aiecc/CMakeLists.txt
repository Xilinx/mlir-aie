#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

llvm_map_components_to_libnames(llvm_libs support core irreader)

# Build aiecc compiler driver
add_llvm_executable(aiecc aiecc.cpp)
llvm_update_compile_flags(aiecc)
install(TARGETS aiecc
  EXPORT AIETargets
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT aiecc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ADF
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransformDialect
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  AIE
  AIETargets
  AIETransforms
  AIEX
  AIEXTransforms
  AIEXUtils
  MLIRAIEToConfiguration
  MLIRAIEVecDialect
  MLIRAIEVecAIE1Dialect
  MLIRAIEVecTransformOps
  MLIRAIEVecTransforms
  MLIRAIEVecToLLVM
  MLIRXLLVMDialect
  ${llvm_libs}
  )

target_link_libraries(aiecc PUBLIC ${LIBS})

# Optional: AIEBU library for direct ELF generation (avoids subprocess to aiebu-asm)
if(aiebu_FOUND)
  target_link_libraries(aiecc PRIVATE AIEBU::aiebu_static)
  target_include_directories(aiecc PRIVATE ${AIEBU_INCLUDE_DIRS})
  target_compile_definitions(aiecc PRIVATE AIECC_HAS_AIEBU_LIBRARY)
  message(STATUS "aiecc: Enabled AIEBU library integration")
endif()

# NOTE: bootgen library integration is not currently supported because
# bootgen uses C++ exceptions for error handling, but LLVM is built with
# -fno-exceptions. The bootgen subprocess fallback is always used instead.
