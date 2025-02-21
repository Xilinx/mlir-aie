# vector_scalar_mul/vector_scalar_mul_alt.py -*- Python -*-
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


def my_vector_scalar(dev, in1_size, in2_size, out_size, trace_size):
    # N = vector_size
    # N_in_bytes = N * 2 # TODO How to force this to match data type
    N_in_bytes = in1_size  # TODO How to force this to match data type
    N = N_in_bytes // 2
    N_div_n = 4  # chop input vector into 4 sub-vectors
    n = N // N_div_n

    assert in2_size == 4, "2nd input buffer must be size 4 (4 bytes = 1 integer)."
    assert out_size == in1_size, "Output buffer size must match input buffer size."

    buffer_depth = 2

    vectorized = True

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[np.int16]]
        tile_ty = np.ndarray[(n,), np.dtype[np.int16]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

        # AIE Core Function declarations

        func_type = "vector" if vectorized else "scalar"
        scale = external_func(
            f"vector_scalar_mul_int16_{func_type}",
            inputs=[tile_ty, tile_ty, scalar_ty, np.int32],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_factor = object_fifo(
            "infactor", ShimTile, ComputeTile2, buffer_depth, scalar_ty
        )
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, tile_ty)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in range_(N_div_n):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale(elem_in, elem_out, elem_factor, n)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):

            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace, ShimTile, trace_size, N_in_bytes
                )

            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, N], issue_token=True
            )
            in_factor_task = shim_dma_single_bd_task(
                of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in_task, in_factor_task, out_task)
            dma_await_task(in_task, in_factor_task, out_task)


try:
    if len(sys.argv) < 5:
        raise ValueError(
            "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
        )

    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_1col
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    in1_size = int(sys.argv[2])
    if in1_size % 128 != 0 or in1_size < 1024:
        print(
            "In1 buffer size must be a multiple of 128 (so len is multiple of 64) and greater than or equal to 1024 (so len >= 512)"
        )
        raise ValueError
    in2_size = int(sys.argv[3])
    out_size = int(sys.argv[4])
    trace_size = 0 if (len(sys.argv) != 6) else int(sys.argv[5])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    my_vector_scalar(dev, in1_size, in2_size, out_size, trace_size)
print(ctx.module)
