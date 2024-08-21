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
    mobilenetV3BottleneckA("bn1",withSkip=False, depthWiseStride=2, tensorInW=112, tensorInH=112, tensorInC=16,tensorOutC=24,depthWiseChannels=64, scaleFactor1=8, scaleFactor2=7, scaleFactor3=9) # bottleneck 1
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

