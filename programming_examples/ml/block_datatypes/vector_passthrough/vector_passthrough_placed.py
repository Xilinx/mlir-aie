# vector_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx


def vector_passthrough():
    N = 32
    n = 16

    buffer_depth = 2

    if len(sys.argv) != 2:
        raise ValueError("[ERROR] Need 1 command line arguments (Col)")

    @device(AIEDevice.npu2)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[v8bfp16ebs8]]
        tile_ty = np.ndarray[(n,), np.dtype[v8bfp16ebs8]]

        kernel_func = external_func("bfp16_passthrough_vectorized", [tile_ty, tile_ty])

        # Tile declarations
        ShimTile = tile(int(sys.argv[1]), 0)
        ComputeTile2 = tile(int(sys.argv[1]), 2)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo("in1", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, tile_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "kernel.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)

                # Kernel call
                kernel_func(elem_in1, elem_out)

                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):
            in1_task = shim_dma_single_bd_task(of_in1, A, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in1_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in1_task)


with mlir_mod_ctx() as ctx:
    vector_passthrough()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
