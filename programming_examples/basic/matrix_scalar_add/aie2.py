# matrix_scalar_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

import sys

# Size of the entire image
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Size of the tile we are processing
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

NUM_3D = IMAGE_WIDTH / TILE_WIDTH
NUM_4D = IMAGE_HEIGHT / TILE_HEIGHT

objfifo_capacity = 4


def my_matrix_add_one():

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        memRef_ty = T.memref(TILE_SIZE, T.i32())

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # AIE-array data movement with object fifos
        # Input
        of_in1 = object_fifo("in0", ShimTile, ComputeTile2, objfifo_capacity, memRef_ty)

        # Output
        of_out1 = object_fifo(
            "out0", ComputeTile2, ShimTile, objfifo_capacity, memRef_ty
        )

        # Set up compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(TILE_SIZE):
                    v0 = memref.load(elem_in, [i])
                    v1 = arith.addi(v0, arith.constant(1, T.i32()))
                    memref.store(v1, elem_out, [i])
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement

        tensor_ty = T.memref(TILE_SIZE, T.i32())

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
            npu_dma_memcpy_nd(
                metadata="out0",
                bd_id=0,
                mem=outTensor,
                sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                strides=[1, 1, IMAGE_WIDTH, 1],
            )
            npu_dma_memcpy_nd(
                metadata="in0",
                bd_id=1,
                mem=inTensor,
                sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                strides=[1, 1, IMAGE_WIDTH, 1],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_matrix_add_one()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
