##===- CMakeLists.txt - convert-phy-to-aie cmake  -------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_conversion_library(PhysicalToAie
  PhysicalToAie.cpp

  DEPENDS
  MLIRPhyConversionPassIncGen

  LINK_LIBS
  AIE
  MLIRPass
  PhyTransformAIE
)

set_property(GLOBAL APPEND PROPERTY PHY_EXPORTS AIE)
