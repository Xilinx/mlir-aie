#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import sys
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def single_buffer():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            data_ty = np.ndarray[(16,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)

            # AIE-array data movement with object fifos
            # Input
            of_in = object_fifo(
                "in", ComputeTile2, ComputeTile3, 1, data_ty
            )  # single buffer
            # Output
            of_out = object_fifo(
                "out", ComputeTile3, ShimTile, 1, data_ty
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
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        elem_out[i] = elem_in[i]
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(data_ty, data_ty, data_ty)
            def sequence(A, B, C):
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, 16], issue_token=True
                )
                dma_start_task(out_task)
                dma_await_task(out_task)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


single_buffer()
