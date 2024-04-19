#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx


def join_L2():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_24_ty = T.memref(24, T.i32())
            memRef_8_ty = T.memref(8, T.i32())

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile0 = tile(0, 2)
            ComputeTile1 = tile(0, 3)
            ComputeTile2 = tile(0, 4)

            # AIE-array data movement with object fifos
            # Output
            of_out = object_fifo("out", MemTile, ShimTile, 2, memRef_24_ty)
            of_out0 = object_fifo("out0", ComputeTile0, MemTile, 2, memRef_8_ty)
            of_out1 = object_fifo("out1", ComputeTile1, MemTile, 2, memRef_8_ty)
            of_out2 = object_fifo("out2", ComputeTile2, MemTile, 2, memRef_8_ty)
            object_fifo_link([of_out0, of_out1, of_out2], of_out)

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile0)
            def core_body():
                # Effective while(1)
                for _ in for_(6):
                    elem = of_out0.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(8):
                        v0 = memref.load(elem, [i])
                        v1 = arith.constant(1, T.i32())
                        memref.store(v1, elem, [i])
                        yield_([])
                    of_out0.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile1)
            def core_body():
                # Effective while(1)
                for _ in for_(6):
                    elem = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(8):
                        v0 = memref.load(elem, [i])
                        v1 = arith.constant(1, T.i32())
                        memref.store(v1, elem, [i])
                        yield_([])
                    of_out1.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(6):
                    elem = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(8):
                        v0 = memref.load(elem, [i])
                        v1 = arith.constant(1, T.i32())
                        memref.store(v1, elem, [i])
                        yield_([])
                    of_out2.release(ObjectFifoPort.Produce, 1)
                    yield_([])

    print(ctx.module)


join_L2()
