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

# Deciphering the command line arguments
if len(sys.argv) < 3:
    raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

if sys.argv[1] == "npu":
    dev = AIEDevice.npu
elif sys.argv[1] == "xcvc1902":
    dev = AIEDevice.xcvc1902
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

col = int(sys.argv[2])


def my_vector_scalar():
    M = 13
    N = 13
    CinUp = 8
    CoutUp = 8
    F = 3
    S = 1
    Mout = M - F // S + 1 
    Nout = N - F // S + 1

    repeats = 1
    input_size = M * N * CinUp #1352
    input_total = input_size // 2

    output_size = Mout * Nout * CoutUp  #7744
    output_total = output_size // 2

    weight_size = F * F * CinUp * CoutUp

    buffer_depth = 1

    vectorized = True
    enable_tracing = False

    if enable_tracing and sys.argv[1] == "xcvc1902":
        raise ValueError(
            "[ERROR] Trace is currently not supported with device xcvc1902"
        )

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(input_size, T.ui8())
            weight_ty = T.memref(weight_size, T.i8())
            out_memRef_ty = T.memref(output_size, T.ui8())
            # AIE Core Function declarations
            conv2d = external_func(
            "conv2d_int8", inputs=[memRef_ty, weight_ty, out_memRef_ty, T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32()]
            )

            # Tile declarations
            ShimTile = tile(col, 0)
            compute_tile2_col, compute_tile2_row = col, 2
            ComputeTile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
            of_weight = object_fifo("weight", ShimTile, ComputeTile2, buffer_depth, weight_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, out_memRef_ty)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "conv2d.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(repeats):
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                        elem_in2 = of_weight.acquire(ObjectFifoPort.Consume, 1)
                        call(conv2d, [elem_in, elem_in2, elem_out, M, N, CinUp, CoutUp, F, S, 0, 0, 0])
                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_weight.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            in_ty = T.memref(input_size, T.i32())
            w_ty = T.memref(weight_size, T.i32())
            out_ty = T.memref(output_size, T.i32())

            @FuncOp.from_py_func(in_ty, w_ty, out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, repeats, output_total])
                npu_dma_memcpy_nd(metadata="in", bd_id=2, mem=A, sizes=[1, 1, repeats, input_total])
                npu_dma_memcpy_nd(metadata="weight", bd_id=1, mem=B, sizes=[1, 1, repeats, weight_size])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
