# vector_scalar_mul/aie2.py -*- Python -*-
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
from aie.extras.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


def my_vector_scalar(vector_size, trace_size):
    N = vector_size
    N_in_bytes = N * 2
    N_div_n = 4  # chop input vector into 4 sub-vectors
    n = N // N_div_n

    buffer_depth = 2

    vectorized = True

    @device(AIEDevice.npu1_1col)
    def device_body():
        tensor_ty = np.ndarray[np.int16, (N,)]
        tile_ty = np.ndarray[np.int16, (n,)]
        scalar_ty = np.ndarray[np.int32, (1,)]

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

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

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
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    ddr_id=2,
                    size=trace_size,
                    offset=N_in_bytes,
                )
            npu_dma_memcpy_nd(
                metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
            )
            npu_dma_memcpy_nd(
                metadata=of_factor, bd_id=2, mem=F, sizes=[1, 1, 1, 1], issue_token=True
            )
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
            dma_wait(of_in, of_factor, of_out)


try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    my_vector_scalar(vector_size, trace_size)
    print(ctx.module)
