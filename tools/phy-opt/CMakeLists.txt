##===- CMakeLists.txt - phy-opt cmake  ------------------------*- cmake -*-===//
##
## Configure the phy-opt build.
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

get_property(dialect_libs GLOBAL PROPERTY PHY_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY PHY_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(phy-opt
  phy-opt.cpp
)

llvm_update_compile_flags(phy-opt)

target_link_libraries(phy-opt
  PRIVATE
  AIE
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
  ${dialect_libs}
  ${conversion_libs}
)
