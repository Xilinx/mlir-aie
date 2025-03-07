# vector_reduce_max/vector_reduce_max_alt.py -*- Python -*-
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

import aie.utils.trace as trace_utils


def vector_reduce_max(dev, in1_size, out_size, trace_size):
    N_in_bytes = in1_size
    N = N_in_bytes // 4
    O_in_bytes = out_size
    O = O_in_bytes // 4

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    buffer_depth = 2

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(N,), np.dtype[np.int32]]
        out_ty = np.ndarray[(O,), np.dtype[np.int32]]

        # AIE Core Function declarations
        reduce_max_vector = external_func(
            "reduce_max_vector", inputs=[in_ty, out_ty, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2, ShimTile]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, in_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, out_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "reduce_max.cc.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                reduce_max_vector(elem_in, elem_out, N)
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, O], issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")

device_name = str(sys.argv[1])
if device_name == "npu":
    dev = AIEDevice.npu1_1col
elif device_name == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(sys.argv[2])
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(sys.argv[3])
trace_size = 0 if (len(sys.argv) != 5) else int(sys.argv[4])

with mlir_mod_ctx() as ctx:
    vector_reduce_max(dev, in1_size, out_size, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
