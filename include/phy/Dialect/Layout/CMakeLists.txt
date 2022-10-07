##===- CMakeLists.txt - layout dialect cmake  -----------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_dialect(Layout layout)

set(LLVM_TARGET_DEFINITIONS Layout.td)
phy_tablegen(LayoutEnums.h.inc -gen-enum-decls)
phy_tablegen(LayoutEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLayoutEnumsIncGen)
add_phy_doc(Layout LayoutDialect -gen-dialect-doc -dialect layout)
add_dependencies(phy-headers MLIRLayoutEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS LayoutPasses.td)
phy_tablegen(LayoutPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRLayoutTransformsIncGen)
add_phy_doc(LayoutPasses LayoutPasses -gen-pass-doc)
add_dependencies(phy-headers MLIRLayoutTransformsIncGen)
