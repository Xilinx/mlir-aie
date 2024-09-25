# memtile_repeat/distribute_repeat/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

dev = AIEDevice.npu1_1col
col = 0
N = 36

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])

assert N % 2 == 0, "N must be even"
repeat_counter = 6
out_size = N * (repeat_counter + 1)


def distribute_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_in_ty = T.memref(N, T.i32())
            memRef_half_ty = T.memref(N // 2, T.i32())
            memRef_out_ty = T.memref(out_size, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile2 = tile(col, 2)
            ComputeTile3 = tile(col, 3)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, memRef_in_ty)
            of_in2 = object_fifo("in2", MemTile, ComputeTile2, 2, memRef_half_ty)
            of_in3 = object_fifo("in3", MemTile, ComputeTile3, 2, memRef_half_ty)
            of_in2.set_memtile_repeat(repeat_counter)
            of_in3.set_memtile_repeat(repeat_counter)
            object_fifo_link(of_in, [of_in2, of_in3], [], [0, N // 2])

            of_out2 = object_fifo("out2", ComputeTile2, MemTile, 2, memRef_half_ty)
            of_out3 = object_fifo("out3", ComputeTile3, MemTile, 2, memRef_half_ty)
            of_out = object_fifo("out", MemTile, ShimTile, 1, memRef_out_ty)
            object_fifo_link([of_out2, of_out3], of_out, [0, out_size // 2], [])

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(N // 2):
                        v0 = memref.load(elemIn, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elemOut, [i])
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                for _ in range_(sys.maxsize):
                    elemOut = of_out3.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in3.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(N // 2):
                        v0 = memref.load(elemIn, [i])
                        v1 = arith.addi(v0, arith.constant(2, T.i32()))
                        memref.store(v1, elemOut, [i])
                    of_in3.release(ObjectFifoPort.Consume, 1)
                    of_out3.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            tensor_out_ty = T.memref(out_size, T.i32())
            tensor_in_ty = T.memref(N, T.i32())

            @runtime_sequence(tensor_in_ty, tensor_in_ty, tensor_out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, out_size]
                )
                dma_wait(of_out)

    print(ctx.module)


distribute_repeat()
