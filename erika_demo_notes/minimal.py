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
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

IMAGE_HEIGHT, IMAGE_WIDTH = 16, 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
TILE_HEIGHT, TILE_WIDTH = 8, 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

with mlir_mod_ctx() as ctx:
    def my_matrix_add_one():
        @device(AIEDevice.npu1_1col)
        def device_body():
            tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            of_in1 = object_fifo("in0", ShimTile, ComputeTile2, 2, tile_ty)
            of_out1 = object_fifo("out0", ComputeTile2, ShimTile, 2, tile_ty)

            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(TILE_SIZE):
                        elem_out[i] = elem_in[i] + 1
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(tile_ty, tile_ty, tile_ty)
            def sequence(inTensor, _, outTensor):
                npu_dma_memcpy_nd(metadata=of_in1, bd_id=1,
                    mem=inTensor,
                    sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                    strides=[1, 1, IMAGE_WIDTH, 1],
                    issue_token=True,
                )

                npu_dma_memcpy_nd(metadata=of_out1, bd_id=0,
                    mem=outTensor,
                    sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
                    strides=[1, 1, IMAGE_WIDTH, 1],
                )
                dma_wait(of_in1, of_out1)
