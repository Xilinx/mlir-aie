##===- CMakeLists.txt - nop pass cmake  -----------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_conversion_library(Nop
  Nop.cpp

  DEPENDS
  PhyConnectivity
  MLIRPhyConversionPassIncGen

  LINK_LIBS
  MLIRPass
)