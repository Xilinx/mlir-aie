# bfp_conversion_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx


def bfp_conversion():
    # We are just doing one operation in total => tensor and tile sizes are equal
    N_in = 64
    N_out = 8
    n_in = 64
    n_out = 8

    buffer_depth = 2

    if len(sys.argv) != 2:
        raise ValueError("[ERROR] Need 1 command line arguments (Col)")

    @device(AIEDevice.npu2)
    def device_body():
        tensor_bf16_ty = np.ndarray[(N_in,), np.dtype[bfloat16]]
        tile_bf16_ty = np.ndarray[(n_in,), np.dtype[bfloat16]]

        tensor_bfp16_ty = np.ndarray[(N_out,), np.dtype[v8bfp16ebs8]]
        tile_bfp16_ty = np.ndarray[(n_out,), np.dtype[v8bfp16ebs8]]

        # AIE Core Function declarations
        conversion_func = external_func(
            "bf16_to_bfp_conversion",
            [tile_bf16_ty, tile_bf16_ty, tile_bfp16_ty, tile_bfp16_ty],
        )

        multiplication_func = external_func(
            "bfp16_matrix_multiplication", [tile_bfp16_ty, tile_bfp16_ty, tile_bfp16_ty]
        )

        # Tile declarations
        ShimTile = tile(int(sys.argv[1]), 0)
        ComputeTile2 = tile(int(sys.argv[1]), 2)
        ComputeTile3 = tile(int(sys.argv[1]), 3)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo("in1", ShimTile, ComputeTile2, buffer_depth, tile_bf16_ty)
        of_in2 = object_fifo("in2", ShimTile, ComputeTile2, buffer_depth, tile_bf16_ty)
        of_intermediate1 = object_fifo(
            "intermediate1", ComputeTile2, ComputeTile3, buffer_depth, tile_bfp16_ty
        )
        of_intermediate2 = object_fifo(
            "intermediate2", ComputeTile2, ComputeTile3, buffer_depth, tile_bfp16_ty
        )
        of_out = object_fifo("out", ComputeTile3, ShimTile, buffer_depth, tile_bfp16_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "kernel.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                elem_out1 = of_intermediate1.acquire(ObjectFifoPort.Produce, 1)
                elem_out2 = of_intermediate2.acquire(ObjectFifoPort.Produce, 1)

                conversion_func(elem_in1, elem_in2, elem_out1, elem_out2)

                of_in1.release(ObjectFifoPort.Consume, 1)
                of_in2.release(ObjectFifoPort.Consume, 1)
                of_intermediate1.release(ObjectFifoPort.Produce, 1)
                of_intermediate2.release(ObjectFifoPort.Produce, 1)

        # Compute tile 3
        @core(ComputeTile3, "kernel.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in1 = of_intermediate1.acquire(ObjectFifoPort.Consume, 1)
                elem_in2 = of_intermediate2.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)

                multiplication_func(elem_in1, elem_in2, elem_out)

                of_intermediate1.release(ObjectFifoPort.Consume, 1)
                of_intermediate2.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_bf16_ty, tensor_bf16_ty, tensor_bfp16_ty)
        def sequence(A, B, C):
            # The first matrix is accepted as is
            in1_task = shim_dma_single_bd_task(of_in1, A, sizes=[1, 1, 1, N_in])
            # Note that properly aligning dot products and bfps implies transposing before converting to bfp!
            # Otherwise, the dot products inside the matrix multiplication will not properly group the blocks
            # The second matrix must be transposed for the multiplication, before the conversion to bfp
            # To perform the transposition at this level, we would need a granularity aligned with 4 bytes.
            # Unfortunately, this is not possible with bf16. Can be done instead inside the core for this simple example.
            in2_task = shim_dma_single_bd_task(of_in2, B, sizes=[1, 1, 1, N_in])

            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, N_out], issue_token=True
            )

            dma_start_task(in1_task, in2_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in1_task, in2_task)


with mlir_mod_ctx() as ctx:
    bfp_conversion()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
