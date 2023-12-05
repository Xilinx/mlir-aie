#
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
from aie.dialects.extras import memref, arith
from aie.util import mlir_mod_ctx


def my_add_one_objFifo():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_16_ty = T.memref(16, T.i32())
            memRef_8_ty = T.memref(8, T.i32())
            ofifo_memRef_16_ty = TypeAttr.get(ObjectFifoType.get(memRef_16_ty))
            ofifo_memRef_8_ty = TypeAttr.get(ObjectFifoType.get(memRef_8_ty))

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input
            objectfifo("in0", ShimTile, [MemTile], 2, ofifo_memRef_16_ty, [], [])
            objectfifo("in1", MemTile, [ComputeTile2], 2, ofifo_memRef_8_ty, [], [])
            objectfifo_link(["in0"], ["in1"])

            # Output
            objectfifo("out0", MemTile, [ShimTile], 2, ofifo_memRef_16_ty, [], [])
            objectfifo("out1", ComputeTile2, [MemTile], 2, ofifo_memRef_8_ty, [], [])
            objectfifo_link(["out1"], ["out0"])

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = acquire(
                        ObjectFifoPort.Consume, "in1", 1, memRef_8_ty
                    ).acquired_elem()
                    elem_out = acquire(
                        ObjectFifoPort.Produce, "out1", 1, memRef_8_ty
                    ).acquired_elem()
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "in1", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "out1", 1)
                    yield_([])

            # To/from AIE-array data movement

            memRef_64_ty = T.memref(64, T.i32())
            memRef_32_ty = T.memref(32, T.i32())

            @FuncOp.from_py_func(memRef_64_ty, memRef_32_ty, memRef_64_ty)
            def sequence(inTensor, notUsed, outTensor):
                ipu_dma_memcpy_nd(
                    metadata="out0", bd_id=0, mem=outTensor, lengths=[1, 1, 1, 64]
                )
                ipu_dma_memcpy_nd(
                    metadata="in0", bd_id=1, mem=inTensor, lengths=[1, 1, 1, 64]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_add_one_objFifo()
