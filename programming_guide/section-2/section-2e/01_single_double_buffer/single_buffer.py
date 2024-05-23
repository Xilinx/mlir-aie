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


def single_buffer():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            memRef_16_ty = T.memref(16, T.i32())

            # Tile declarations
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)

            # AIE-array data movement with object fifos
            # Input
            of_in = object_fifo(
                "in", ComputeTile2, ComputeTile3, 1, memRef_16_ty
            )  # single buffer

            # Set up compute tiles
            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_out = of_in.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(16):
                        v1 = arith.constant(1, T.i32())
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    yield_([])

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


single_buffer()
