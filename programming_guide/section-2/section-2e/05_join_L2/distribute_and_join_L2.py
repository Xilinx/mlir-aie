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


def distribute_join_L2():
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

            # Output
            of_out = object_fifo("out", MemTile, ShimTile, 2, tile24_ty)
            of_out0 = object_fifo("out0", ComputeTile0, MemTile, 2, tile8_ty)
            of_out1 = object_fifo("out1", ComputeTile1, MemTile, 2, tile8_ty)
            of_out2 = object_fifo("out2", ComputeTile2, MemTile, 2, tile8_ty)
            object_fifo_link([of_out0, of_out1, of_out2], of_out, [0, 8, 16], [])

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile0)
            def core_body():
                # Effective while(1)
                for _ in range_(2):
                    elem_in = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out0.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        elem_out[i] = elem_in[i] + 1
                    of_in0.release(ObjectFifoPort.Consume, 1)
                    of_out0.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile1)
            def core_body():
                # Effective while(1)
                for _ in range_(2):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        elem_out[i] = elem_in[i] + 1
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            # Compute tile 4
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in range_(2):
                    elem_in = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        elem_out[i] = elem_in[i] + 1
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce, 1)

            data_ty = np.ndarray[(48,), np.dtype[np.int32]]

            @runtime_sequence(data_ty, data_ty, data_ty)
            def sequence(inTensor, notUsed, outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=1,
                    mem=inTensor,
                    sizes=[1, 1, 1, 48],
                    issue_token=True,
                )
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 48]
                )
                dma_wait(of_in, of_out)

    print(ctx.module)


distribute_join_L2()
