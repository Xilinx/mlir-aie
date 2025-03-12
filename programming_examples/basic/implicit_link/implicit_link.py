# implicit_link.py -*- Python -*-
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
from aie.dialects.aie import tile, object_fifo, core, ObjectFifoPort
from aie.helpers.dialects.ext.scf import _for as range_

N = 4096
dev = AIEDevice.npu1_1col
col = 0
line_size = 1024
op_size = 512

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def implicit_link():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]
            op_ty = np.ndarray[(op_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)
            ComputeTile3 = tile(col, 3)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, {ComputeTile2, ComputeTile3}, 2, [line_ty, op_ty], [], [0, op_size])  #Number of objects     # 2 changes: array of datatypes, srcOffsets and dstOffsets
            of_out = object_fifo("out", {ComputeTile2, ComputeTile3}, ShimTile, 2, [op_ty, line_ty], [0, op_size], [])

            # Set up compute tiles
            def compute_core(tile):                             # Now uses only of_in and of_out in both computeTiles but needs differentiation inside
                @core(tile)
                def core_body():
                    for _ in range_(sys.maxsize):
                        # Add 1 to the input data
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        for i in range_(op_size):
                            elem_out[i] = elem_in[i] + 1
                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)

            compute_core(ComputeTile2)
            compute_core(ComputeTile3)

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                in_task = shim_dma_single_bd_task(
                    of_in, A, sizes=[1, 1, 1, N], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    print(ctx.module)


implicit_link()
