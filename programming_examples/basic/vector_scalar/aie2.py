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
    N = 4096
    N_in_bytes = N * 4
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    vectorized = True
    enable_tracing = False

    if enable_tracing and sys.argv[1] == "xcvc1902":
        raise ValueError(
            "[ERROR] Trace is currently not supported with device xcvc1902"
        )

    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(n, T.i32())
            memRef_ty2 = T.memref(1, T.i32())

            # AIE Core Function declarations
            scale_scalar_int32 = external_func(
                "scale_int32", inputs=[memRef_ty, memRef_ty]
            )
            scale_int32 = external_func("scale_scalar_int32", inputs=[memRef_ty, memRef_ty])

            # Tile declarations
            ShimTile = tile(col, 0)
            compute_tile2_col, compute_tile2_row = col, 2
            ComputeTile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "scale.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                        if vectorized:
                            call(scale_int32, [elem_in, elem_out])
                        else:
                            call(scale_scalar_int32, [elem_in, elem_out])
                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, F, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
