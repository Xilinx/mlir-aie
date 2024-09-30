# vector_reduce_min/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

import sys


def my_reduce_min():
    N = 1024

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        memRef_I_ty = T.memref(N, T.i32())
        memRef_O_ty = T.memref(1, T.i32())

        # AIE Core Function declarations
        reduce_min_vector = external_func(
            "reduce_min_vector", inputs=[memRef_I_ty, memRef_O_ty, T.i32()]
        )

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_I_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_O_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "reduce_min.cc.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                reduce_min_vector(elem_in, elem_out, N)
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.i32())

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):
            npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, 1])
            # of_out will only complete after of_in completes, so we just wait on of_out instead of both
            dma_wait(of_out)


with mlir_mod_ctx() as ctx:
    my_reduce_min()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
