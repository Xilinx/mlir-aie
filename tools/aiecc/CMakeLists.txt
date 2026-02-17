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
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
  AIE
  AIEX
  )

target_link_libraries(aiecc PUBLIC ${LIBS})
