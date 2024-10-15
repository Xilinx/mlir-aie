# dynamic_object_fifo/sliding_window/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N = 100
n_rows = 10
dev = AIEDevice.npu1_1col
col = 0


def sliding_window():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            subtensor_ty = np.ndarray[(N // n_rows,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile, 3, subtensor_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, subtensor_ty)

            # AIE Core Function declarations
            add_10_i32 = external_func(
                "add_10_i32", inputs=[subtensor_ty, subtensor_ty, subtensor_ty]
            )

            # Set up compute tiles

            @core(ComputeTile, "kernel.o")
            def core_body():
                elemOutPre = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemInPre = of_in.acquire(ObjectFifoPort.Consume, 1)
                add_10_i32(elemInPre, elemInPre, elemOutPre)
                of_out.release(ObjectFifoPort.Produce, 1)

                for _ in range_(8):
                    elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elemsIn = of_in.acquire(ObjectFifoPort.Consume, 2)
                    add_10_i32(elemsIn[0], elemsIn[1], elemOut)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

                elemOutPost = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemsInPost = of_in.acquire(ObjectFifoPort.Consume, 2)
                add_10_i32(elemsInPost[0], elemsInPost[1], elemOutPost)
                of_in.release(ObjectFifoPort.Consume, 2)
                of_out.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
                )
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
                dma_wait(of_in, of_out)

    print(ctx.module)


sliding_window()
