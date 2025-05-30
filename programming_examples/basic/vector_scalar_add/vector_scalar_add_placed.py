# vector_scalar_add/vector_scalar_add_placed.py -*- Python -*-
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

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def my_vector_bias_add():
    @device(dev)
    def device_body():
        mem_tile_ty = np.ndarray[(MEM_TILE_WIDTH,), np.dtype[np.int32]]
        aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
        all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        # Input
        of_in0 = object_fifo("in0", ShimTile, MemTile, 2, mem_tile_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, aie_tile_ty)
        object_fifo_link(of_in0, of_in1)

        # Output
        of_out0 = object_fifo("out0", MemTile, ShimTile, 2, mem_tile_ty)
        of_out1 = object_fifo("out1", ComputeTile2, MemTile, 2, aie_tile_ty)
        object_fifo_link(of_out1, of_out0)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(AIE_TILE_WIDTH):
                    elem_out[i] = elem_in[i] + 1
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(all_data_ty, all_data_ty)
        def sequence(inTensor, outTensor):
            in_task = shim_dma_single_bd_task(
                of_in0, inTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out0, outTensor, sizes=[1, 1, 1, PROBLEM_SIZE], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    my_vector_bias_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
