# memtile_repeat/ping_pong/aie2.py -*- Python -*-
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
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_1col
col = 0

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


def ping_pong():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(1024, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, memRef_ty)
            of_out = object_fifo("out", MemTile, ShimTile, [2, 1], memRef_ty)
            of_out.set_memtile_repeat(4)
            object_fifo_link(of_in, of_out)

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())
            tensor_in_ty = T.memref(N // 4, T.i32())

            @FuncOp.from_py_func(tensor_in_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N // 4])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


ping_pong()
