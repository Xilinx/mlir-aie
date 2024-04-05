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

# Used to get command line arguments
import sys


def my_add_one_objFifo():
    with mlir_mod_ctx() as ctx:

        if len(sys.argv) != 3:
            raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

        if sys.argv[1] == "ipu":
            dev = AIEDevice.ipu
        elif sys.argv[1] == "xcvc1902":
            dev = AIEDevice.xcvc1902
        else:
            raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

        @device(dev)
        def device_body():
            memRef_16_ty = T.memref(16, T.i32())
            memRef_8_ty = T.memref(8, T.i32())

            # Tile declarations
            ShimTile = tile(int(sys.argv[2]), 0)
            ComputeTile2 = tile(int(sys.argv[2]), 2)

            # AIE-array data movement with object fifos
            # Input
            of_in1 = object_fifo("in0", ShimTile, ComputeTile2, 2, memRef_16_ty)

            # Output
            of_out1 = object_fifo("out0", ComputeTile2, ShimTile, 2, memRef_16_ty)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(16):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement

            memRef_64_ty = T.memref(64, T.i32())
            memRef_32_ty = T.memref(32, T.i32())

            @FuncOp.from_py_func(memRef_64_ty, memRef_32_ty, memRef_64_ty)
            def sequence(inTensor, notUsed, outTensor):
                ipu_dma_memcpy_nd(
                    metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, 64]
                )
                ipu_dma_memcpy_nd(
                    metadata="in0", bd_id=1, mem=inTensor, sizes=[1, 1, 1, 64]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_add_one_objFifo()
