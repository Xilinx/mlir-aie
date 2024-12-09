# matrix_scalar_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
# fmt: off
import numpy as np
import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

# Size of the entire matrix
MATRIX_HEIGHT = 16
MATRIX_WIDTH = 128

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16

with mlir_mod_ctx() as ctx:
  @device(AIEDevice.npu1_1col)
  def device_body():
    # Types, tile declarations, and AIE data movement with object fifos
    matrix_ty = np.ndarray[(MATRIX_HEIGHT, MATRIX_WIDTH), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE_HEIGHT, TILE_WIDTH), np.dtype[np.int32]]
    ShimTile = tile(0, 0)
    ComputeTile2 = tile(0, 2)
    of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
    of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

    @core(ComputeTile2) # Set up compute tile 2
    def core_body():
      for _ in range_(sys.maxsize): # Effective while(1)
        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
        for i in range_(TILE_HEIGHT):
          for j in range_(TILE_WIDTH):
            elem_out[i, j] = elem_in[i, j] + 1
        of_in.release(ObjectFifoPort.Consume, 1)
        of_out.release(ObjectFifoPort.Produce, 1)

    @runtime_sequence(matrix_ty, matrix_ty)
    def sequence(inTensor, outTensor):
      npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=inTensor,
          sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[0, 0, MATRIX_WIDTH, 1])
      npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=outTensor,
          sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[0, 0, MATRIX_WIDTH, 1])
      dma_wait(of_out)

  print(ctx.module) # Emit MLIR strings

# fmt: on
