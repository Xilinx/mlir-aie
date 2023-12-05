# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.util import mlir_mod_ctx


def printf():
    N = 512

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(N, T.i32())
            ofifo_memRef_ty = TypeAttr.get(ObjectFifoType.get(memRef_ty))

            # AIE Core Function declarations
            kernel = external_func("kernel", inputs=[memRef_ty, memRef_ty, memRef_ty])

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            objectfifo("inOF", ShimTile, [ComputeTile2], 2, ofifo_memRef_ty, [], [])
            objectfifo("outOF", ComputeTile2, [ShimTile], 2, ofifo_memRef_ty, [], [])
            objectfifo("logoutOF", ComputeTile2, [ShimTile], 2, ofifo_memRef_ty, [], [])

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "kernel.o")
            def core_body():
                elemOut = acquire(
                    ObjectFifoPort.Produce, "outOF", 1, memRef_ty
                ).acquired_elem()
                elemIn = acquire(
                    ObjectFifoPort.Consume, "inOF", 1, memRef_ty
                ).acquired_elem()
                elemLogout = acquire(
                    ObjectFifoPort.Produce, "logoutOF", 1, memRef_ty
                ).acquired_elem()
                Call(kernel, [elemIn, elemOut, elemLogout])
                objectfifo_release(ObjectFifoPort.Consume, "inOF", 1)
                objectfifo_release(ObjectFifoPort.Produce, "outOF", 1)
                objectfifo_release(ObjectFifoPort.Produce, "logoutOF", 1)

            # To/from AIE-array data movement
            @FuncOp.from_py_func(memRef_ty, memRef_ty, memRef_ty)
            def sequence(in_mem, out_mem, logout):
                ipu_dma_memcpy_nd(
                    metadata="outOF", bd_id=0, mem=out_mem, lengths=[1, 1, 1, N]
                )
                ipu_dma_memcpy_nd(
                    metadata="inOF", bd_id=1, mem=in_mem, lengths=[1, 1, 1, N]
                )
                ipu_dma_memcpy_nd(
                    metadata="logoutOF", bd_id=2, mem=logout, lengths=[1, 1, 1, N]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


printf()
