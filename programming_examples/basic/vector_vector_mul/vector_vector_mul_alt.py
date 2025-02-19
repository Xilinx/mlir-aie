# vector_vector_mul/vector_vector_mul_alt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


def my_vector_mul():
    N = 256
    n = 16
    N_div_n = N // n

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

        # AIE Core Function declarations

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo("in1", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_in2 = object_fifo("in2", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, tile_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(n):
                        elem_out[i] = elem_in1[i] * elem_in2[i]
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            in1_task = shim_dma_single_bd_task(of_in1, A, sizes=[1, 1, 1, N])
            in2_task = shim_dma_single_bd_task(of_in2, B, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in1_task, in2_task, out_task)
            # out_task will only complete after in1_task and in2_task completes, so we just wait on of_out instead of all
            dma_await_task(out_task)
            dma_free_task(in1_task, in2_task)


with mlir_mod_ctx() as ctx:
    my_vector_mul()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
