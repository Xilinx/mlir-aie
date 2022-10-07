##===- CMakeLists.txt - AIE transform cmake  ------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

file(GLOB_RECURSE TransformAIE_Src
  "*.cpp"
)

add_phy_library(PhyTransformAIE
  ${TransformAIE_Src}
  
  DEPENDS
  AIE

  LINK_LIBS
  PhyConnectivity
  PhyRewrite
)
