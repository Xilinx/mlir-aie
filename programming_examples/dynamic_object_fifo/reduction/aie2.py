# dynamic_object_fifo/reduction/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

N = 100
O = 50
n_rows = 10
dev = AIEDevice.npu1_1col
col = 0

def reduction():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(N // n_rows, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile, [2, 4], memRef_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, memRef_ty)

            # AIE Core Function declarations
            add_10_i32 = external_func(
                "add_10_i32", inputs=[memRef_ty, memRef_ty, memRef_ty]
            )

            # Set up compute tiles

            @core(ComputeTile, "kernel.o")
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elemsIn = of_in.acquire(ObjectFifoPort.Consume, 2)
                    call(add_10_i32, [elemsIn[0], elemsIn[1], elemOut])
                    of_in.release(ObjectFifoPort.Consume, 2)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            tensor_in_ty = T.memref(N, T.i32())
            tensor_out_ty = T.memref(O, T.i32())

            @runtime_sequence(tensor_in_ty, tensor_out_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, O])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


reduction()
