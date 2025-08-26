#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneck8And9Static import mobilenetV3Bottleneck8And9

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx

# NOLF: set tensorInH=6 to avoid PM overflow and need for dynamic objFIFO during debug and reduced channels with //2

with mlir_mod_ctx() as ctx:
    mobilenetV3Bottleneck8And9(tensorInW=14, tensorInH=14, tensorInC=80,tensorOutC=80, 
                               bn8_depthWiseChannels = 184, 
                               bn9_depthWiseStride=1, bn9_depthWiseChannels = 184,
                               scaleFactor8_1=9, scaleFactor8_2=8,scaleFactor8_3=10, scaleFactorAdd8=0,
                               scaleFactor9_1=9, scaleFactor9_2=7,scaleFactor9_3=11, scaleFactorAdd9=1)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

