# matrix_scalar_add/matrix_scalar_add_placed.py -*- Python -*-
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
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

# Size of the entire matrix
MATRIX_HEIGHT = 16
MATRIX_WIDTH = 128
MATRIX_SHAPE = (MATRIX_HEIGHT, MATRIX_WIDTH)

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SHAPE = (TILE_HEIGHT, TILE_WIDTH)


def my_matrix_add_one():
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
        # Define tensor types
        matrix_ty = np.ndarray[MATRIX_SHAPE, np.dtype[np.int32]]
        tile_ty = np.ndarray[TILE_SHAPE, np.dtype[np.int32]]

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

        # Set up compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(TILE_HEIGHT):
                    for j in range_(TILE_WIDTH):
                        elem_out[i, j] = elem_in[i, j] + 1
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tap = TensorTiler2D.simple_tiler(MATRIX_SHAPE, TILE_SHAPE)[0]

        @runtime_sequence(matrix_ty, matrix_ty, matrix_ty)
        def sequence(inTensor, _, outTensor):
            in_task = shim_dma_single_bd_task(
                of_in, inTensor, tap=tap, issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, outTensor, tap=tap, issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


with mlir_mod_ctx() as ctx:
    my_matrix_add_one()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
