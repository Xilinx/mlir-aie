#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 64

def my_add_one_objFifo():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.xcvc1902)
        def device_body():
            memRef_ty = T.memref(16, T.i32())

            # Tile declarations
            ShimTile = tile(6, 0)
            ComputeTile2 = tile(6, 2)

            # AIE-array data movement with object fifos
            # Input
            of_in1 = object_fifo("in0", ShimTile, ComputeTile2, 2, memRef_ty)

            # Output
            of_out1 = object_fifo("out0", ComputeTile2, ShimTile, 2, memRef_ty)
            object_fifo_link(of_in1, of_out1)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                tmp = memref.alloc(1, T.i32())
                v0 = arith.constant(0, T.i32())
                memref.store(v0, tmp, [0])

           # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(inTensor, notUsed, outTensor):
                ipu_dma_memcpy_nd(
                    metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, N]
                )
                ipu_dma_memcpy_nd(
                    metadata="in0", bd_id=1, mem=inTensor, sizes=[1, 1, 1, N]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_add_one_objFifo()
