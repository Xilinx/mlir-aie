#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 4096


# Deciphering the command line arguments
if len(sys.argv) < 3:
    raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

if len(sys.argv) == 4:
    N = int(sys.argv[1])

if sys.argv[1] == "npu":
    dev = AIEDevice.npu
elif sys.argv[1] == "xcvc1902":
    dev = AIEDevice.xcvc1902
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

col = int(sys.argv[2])


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(1024, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in for_(sys.maxsize):
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_passthrough()
