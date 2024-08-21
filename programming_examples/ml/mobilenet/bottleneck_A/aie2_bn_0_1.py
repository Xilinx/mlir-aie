#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneck0And1 import mobilenetV3Bottleneck0And1

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx

with mlir_mod_ctx() as ctx:
    mobilenetV3Bottleneck0And1(tensorInW=112, tensorInH=112, tensorInC=16,tensorOutC=24, depthWiseStride=2, depthWiseChannels=64,\
                               scaleFactor0_2=9,scaleFactor0_3=8,scaleFactorAdd0=2,\
                                scaleFactor1_1=8,scaleFactor1_2=8,scaleFactor1_3=9)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

