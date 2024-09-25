# memtile_repeat/simple_repeat/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_1col
col = 0
memtile_repeat_count = 3

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def simple_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(N, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, memRef_ty)
            of_out = object_fifo("out", MemTile, ShimTile, 1, memRef_ty)
            of_out.set_memtile_repeat(memtile_repeat_count)
            object_fifo_link(of_in, of_out)

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())
            tensor_out_ty = T.memref(N * (memtile_repeat_count + 1), T.i32())

            @runtime_sequence(tensor_ty, tensor_ty, tensor_out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=0,
                    mem=C,
                    sizes=[1, 1, 1, N * (memtile_repeat_count + 1)],
                )
                dma_wait(of_out)

    print(ctx.module)


simple_repeat()
