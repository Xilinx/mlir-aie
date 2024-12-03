# matrix_scalar_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aie import (
    tile,
    object_fifo,
    for_,
    core,
    T,
    AIEDevice,
    device,
    ObjectFifoPort,
    yield_,
    runtime_sequence,
    npu_dma_memcpy_nd,
    dma_wait,
)
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D

# Size of the entire image to process
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        # Types, tile declarations, and AIE data movement with object fifos
        tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

        @core(ComputeTile2)  # Set up compute tile 2
        def core_body():
            for _ in range_(sys.maxsize):  # Effective while(1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(TILE_SIZE):
                    elem_out[i] = elem_in[i] + 1
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(tile_ty, tile_ty)
        def sequence(inTensor, outTensor):
            npu_dma_memcpy_nd(
                metadata=of_in,
                bd_id=1,
                mem=inTensor,
                sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                strides=[1, 1, IMAGE_WIDTH, 1],
            )
            npu_dma_memcpy_nd(
                metadata=of_out,
                bd_id=0,
                mem=outTensor,
                sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                strides=[1, 1, IMAGE_WIDTH, 1],
            )
            dma_wait(of_out)

    print(ctx.module)  # Emit MLIR strings
