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
            of_in0 = object_fifo("in0", ShimTile, MemTile, 2, memRef_32_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, memRef_8_ty)
            of_in2 = object_fifo("in2", MemTile, ComputeTile3, 2, memRef_8_ty)
            of_in3 = object_fifo("in3", MemTile, ComputeTile4, 2, memRef_8_ty)
            of_in4 = object_fifo("in4", MemTile, ComputeTile5, 2, memRef_8_ty)
            object_fifo_link(of_in0, [of_in1, of_in2, of_in3, of_in4])

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
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

            # TODO: add the trace
            # TODO: add drawing

    print(ctx.module)


join_L2()
