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

import aie.utils.trace as trace_utils


def my_vector_scalar():

    enable_tracing = False
    trace_size = 8192

    @device(AIEDevice.ipu)
    def device_body():
        memRef_ty = T.memref(1024, T.i32())

        # AIE Core Function declarations
        scale = external_func(
            "scale_scalar_int32", inputs=[memRef_ty, memRef_ty, T.i32(), T.i32()]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in for_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                for _ in for_(4):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    call(scale, [elem_in, elem_out, 1024])
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                yield_([])

        # To/from AIE-array data movement
        tensor_ty = T.memref(4096, T.i32())

        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, C, notUsed):
            ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, 4096])
            ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, 4096])
            ipu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_vector_scalar()
    print(ctx.module)
