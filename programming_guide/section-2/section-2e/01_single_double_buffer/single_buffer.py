#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx


def single_buffer():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            data_ty = np.ndarray[(16,), np.dtype[np.int32]]

            # Tile declarations
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)

            # AIE-array data movement with object fifos
            # Input
            of_in = object_fifo(
                "in", ComputeTile2, ComputeTile3, 1, data_ty
            )  # single buffer

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in range_(8):
                    elem_out = of_in.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        elem_out[i] = 1
                    of_in.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                # Effective while(1)
                for _ in range_(8):
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    of_in.release(ObjectFifoPort.Consume, 1)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


single_buffer()
