#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx


def distribute_L2():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
            tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile0 = tile(0, 2)
            ComputeTile1 = tile(0, 3)
            ComputeTile2 = tile(0, 4)

            # AIE-array data movement with object fifos
            # Input
            of_in = object_fifo("in", ShimTile, MemTile, 2, tile24_ty)
            of_in0 = object_fifo("in0", MemTile, ComputeTile0, 2, tile8_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile1, 2, tile8_ty)
            of_in2 = object_fifo("in2", MemTile, ComputeTile2, 2, tile8_ty)
            object_fifo_link(of_in, [of_in0, of_in1, of_in2], [], [0, 8, 16])

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile0)
            def core_body():
                # Effective while(1)
                for _ in range_(8):
                    elem = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(8):
                        elem[i] = elem[i] + 1
                    of_in0.release(ObjectFifoPort.Consume, 1)

            # Compute tile 3
            @core(ComputeTile1)
            def core_body():
                # Effective while(1)
                for _ in range_(8):
                    elem = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(8):
                        elem[i] = elem[i] + 1
                    of_in1.release(ObjectFifoPort.Consume, 1)

            # Compute tile 4
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in range_(8):
                    elem = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(8):
                        elem[i] = elem[i] + 1
                    of_in2.release(ObjectFifoPort.Consume, 1)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


distribute_L2()
