#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils


def my_vector_scalar(vector_size, trace_size):
    N = vector_size
    N_in_bytes = N * 4
    N_div_n = 4  # chop input vector into 4 sub-vectors
    n = N // N_div_n

    buffer_depth = 2

    vectorized = True

    @device(AIEDevice.npu)
    def device_body():
        memRef_ty = T.memref(n, T.i32())
        memRef_ty2 = T.memref(1, T.i32())

        # AIE Core Function declarations

        scale_scalar = external_func(
            "vector_scalar_mul_aie_scalar",
            inputs=[memRef_ty, memRef_ty, memRef_ty2, T.i32()],
        )
        scale = external_func(
            "vector_scalar_mul_aie", inputs=[memRef_ty, memRef_ty, memRef_ty2, T.i32()]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
        of_factor = object_fifo(
            "infactor", ShimTile, ComputeTile2, buffer_depth, memRef_ty2
        )
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in for_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in for_(N_div_n):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    if vectorized:
                        call(scale, [elem_in, elem_out, elem_factor, n])
                    else:
                        call(scale_scalar, [elem_in, elem_out, elem_factor, n])
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                of_factor.release(ObjectFifoPort.Consume, 1)
                yield_([])

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.i32())
        scalar_ty = T.memref(1, T.i32())

        @FuncOp.from_py_func(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):

            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    ddr_id=2,
                    size=trace_size,
                    offset=N_in_bytes,
                )
            npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="infactor", bd_id=2, mem=F, sizes=[1, 1, 1, 1])
            npu_sync(column=0, row=0, direction=0, channel=0)


try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size <= 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    my_vector_scalar(vector_size, trace_size)
    print(ctx.module)
