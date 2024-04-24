#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import memref, arith

import sys


def my_vector_sum():
    N = 64

    buffer_depth = 2

    with mlir_mod_ctx() as ctx:

        if len(sys.argv) != 3:
            raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

        if sys.argv[1] == "npu":
            dev = AIEDevice.npu
        elif sys.argv[1] == "xcvc1902":
            dev = AIEDevice.xcvc1902
        else:
            raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

        @device(dev)
        def device_body():
            memRef_ty = T.memref(N, T.i32())

            # AIE Core Function declarations

            # Tile declarations
            ShimTile = tile(int(sys.argv[2]), 0)
            ComputeTile2 = tile(int(sys.argv[2]), 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                sum_val = memref.alloc(1, T.i32())
                memref.store(arith.constant(0, T.i32()), sum_val, [0])
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(N):
                        v0 = memref.load(elem_in, [i])
                        v1 = memref.load(sum_val, [0])
                        v2 = arith.addi(v1, v0)
                        memref.store(v2, sum_val, [0])
                        yield_([])

                    v3 = memref.load(sum_val, [0])
                    memref.store(v3, elem_out, [0])
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, 1])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_sum()
