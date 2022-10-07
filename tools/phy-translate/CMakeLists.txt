##===- CMakeLists.txt - phy-translate cmake  ------------------*- cmake -*-===//
##
## Configure the phy-translate build.
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

get_property(dialect_libs GLOBAL PROPERTY PHY_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY PHY_TRANSLATION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(phy-translate
  phy-translate.cpp
)

llvm_update_compile_flags(phy-translate)

target_link_libraries(phy-translate
  PRIVATE
  AIE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
