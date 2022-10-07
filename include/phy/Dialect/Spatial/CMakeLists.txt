##===- CMakeLists.txt - spatial dialect cmake  -----------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_dialect(Spatial spatial)

set(LLVM_TARGET_DEFINITIONS Spatial.td)
phy_tablegen(SpatialEnums.h.inc -gen-enum-decls)
phy_tablegen(SpatialEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSpatialEnumsIncGen)
add_phy_doc(Spatial SpatialDialect -gen-dialect-doc -dialect spatial)
add_dependencies(phy-headers MLIRSpatialEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SpatialPasses.td)
phy_tablegen(SpatialPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRSpatialTransformsIncGen)
add_phy_doc(SpatialPasses SpatialPasses -gen-pass-doc)
add_dependencies(phy-headers MLIRSpatialTransformsIncGen)
