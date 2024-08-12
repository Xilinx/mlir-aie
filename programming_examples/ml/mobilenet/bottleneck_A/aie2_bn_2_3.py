#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneck2And3 import mobilenetV3Bottleneck2And3

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx

with mlir_mod_ctx() as ctx:
    mobilenetV3Bottleneck2And3(tensorInW=56, tensorInH=56, tensorInC=24,tensorOutC=40, 
                                bn2_depthWiseChannels = 72, 
                               bn3_depthWiseStride=2, bn3_depthWiseChannels = 72,
                               scaleFactor0_1=8, scaleFactor0_2=8,scaleFactor0_3=11,scaleFactorAdd0=0,
                               scaleFactor1_1=8,scaleFactor1_2=8,scaleFactor1_3=11)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

