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
            memRef_32_ty = T.memref(32, T.i32())
            memRef_8_ty = T.memref(8, T.i32())

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            # Input
            of_out0 = object_fifo("out0", MemTile, ShimTile, 2, memRef_32_ty)
            of_out1 = object_fifo("out1", ComputeTile2, MemTile, 2, memRef_8_ty)
            of_out2 = object_fifo("out2", ComputeTile3, MemTile, 2, memRef_8_ty)
            of_out3 = object_fifo("out3", ComputeTile4, MemTile, 2, memRef_8_ty)
            of_out4 = object_fifo("out4", ComputeTile5, MemTile, 2, memRef_8_ty)
            object_fifo_loutk([of_out1, of_out2, of_out3, of_out4], of_out0)

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_out1.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    yield_([])

            # Compute tile 4
            @core(ComputeTile4)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in3.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in3.release(ObjectFifoPort.Consume, 1)
                    yield_([])

            # Compute tile 5
            @core(ComputeTile5)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in4.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in4.release(ObjectFifoPort.Consume, 1)
                    yield_([])

    print(ctx.module)


join_L2()
