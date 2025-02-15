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


def external_mem_to_core_L2():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
            tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input
            of_in0 = object_fifo("in0", ShimTile, MemTile, 2, tile24_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, tile8_ty)
            object_fifo_link(of_in0, of_in1)

            # Output
            of_out0 = object_fifo("out0", MemTile, ShimTile, 2, tile24_ty)
            of_out1 = object_fifo("out1", ComputeTile2, MemTile, 2, tile8_ty)
            object_fifo_link(of_out1, of_out0)

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in range_(6):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(8):
                        elem_out[i] = elem_in[i] + 1
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            data_ty = np.ndarray[(48,), np.dtype[np.int32]]

            # To/from AIE-array data movement
            @runtime_sequence(data_ty, data_ty, data_ty)
            def sequence(inTensor, notUsed, outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_in0, bd_id=1, mem=inTensor, sizes=[1, 1, 1, 48]
                )
                npu_dma_memcpy_nd(
                    metadata=of_out0, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 48]
                )
                # of_out0 will only complete after of_in0 completes, so we just wait on of_out0 instead of both
                dma_wait(of_out0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core_L2()
