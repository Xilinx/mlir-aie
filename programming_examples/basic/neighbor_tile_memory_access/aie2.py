# neighbor_tile_memory_access/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# Adapted from vector_scalar_add/aie2.py but with link between ComputeTiles

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

import sys

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32


def my_vector_bias_add():
    @device(AIEDevice.npu1_1col)
    def device_body():
        memRef_aie_tile_ty = T.memref(AIE_TILE_WIDTH, T.i32())

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 4)

        # AIE-array data movement with object fifos
        # Input
        of_in0 = object_fifo("in0", ShimTile, ComputeTile2, 2, memRef_aie_tile_ty)
        of_in1 = object_fifo("in1", ComputeTile2, ComputeTile3, 2, memRef_aie_tile_ty)
        object_fifo_link(of_in0, of_in1)

        # Output
        of_out0 = object_fifo("out0", ComputeTile3, ShimTile, 2, memRef_aie_tile_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile3)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out0.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(AIE_TILE_WIDTH):
                    elem_out[i] = elem_in[i] + 1
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out0.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = T.memref(PROBLEM_SIZE, T.i32())

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=of_in0,
                bd_id=1,
                mem=inTensor,
                sizes=[1, 1, 1, PROBLEM_SIZE],
                issue_token=True,
            )
            npu_dma_memcpy_nd(
                metadata=of_out0, bd_id=0, mem=outTensor, sizes=[1, 1, 1, PROBLEM_SIZE]
            )
            dma_wait(of_in0, of_out0)


# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    my_vector_bias_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
