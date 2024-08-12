#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneckA import mobilenetV3BottleneckA

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx

with mlir_mod_ctx() as ctx:
    mobilenetV3BottleneckA("bn9", withSkip=True, depthWiseStride=1, tensorInW=14, tensorInH=14 ,tensorInC=80,tensorOutC=80,depthWiseChannels=184, scaleFactor1=9, scaleFactor2=8, scaleFactor3=11, scaleFactorAdd=0) # bottleneck 7
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

