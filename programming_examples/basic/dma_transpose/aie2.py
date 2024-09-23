# dma_transpose/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

N = 4096
M = 64
K = 64

if len(sys.argv) == 3:
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = M * K


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            memRef_ty = T.memref(M, K, T.i32())

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                # The strides below are configured to read across all rows in the same column
                # Stride of K in dim/wrap 2 skips an entire row to read a full column
                npu_dma_memcpy_nd(
                    metadata="in",
                    bd_id=1,
                    mem=A,
                    sizes=[1, K, M, 1],
                    strides=[1, 1, K, 1],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_passthrough()
