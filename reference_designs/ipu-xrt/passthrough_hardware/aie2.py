#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.extras import memref, arith
from aie.util import mlir_mod_ctx

N = 4096
N_in_bytes = N * 4

if len(sys.argv) == 2:
    N = int(sys.argv[1])


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(1024, T.i32())
            ofifo_memRef_ty = TypeAttr.get(ObjectFifoType.get(memRef_ty))

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            objectfifo("in", ShimTile, [ComputeTile2], 2, ofifo_memRef_ty, [], [])
            objectfifo("out", ComputeTile2, [ShimTile], 2, ofifo_memRef_ty, [], [])
            objectfifo_link(["in"], ["out"])

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                tmp = memref.alloc([1], T.i32())
                v0 = arith.constant(0, T.i32())
                memref.store(v0, tmp, [0])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_passthrough()
