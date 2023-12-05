#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.util import mlir_mod_ctx


def my_vector_scalar():
    N = 4096
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(n, T.i32())
            ofifo_memRef_ty = TypeAttr.get(ObjectFifoType.get(memRef_ty))

            # AIE Core Function declarations
            scale_int32 = external_func("scale_int32", inputs=[memRef_ty, memRef_ty])

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            objectfifo(
                "in", ShimTile, [ComputeTile2], buffer_depth, ofifo_memRef_ty, [], []
            )
            objectfifo(
                "out", ComputeTile2, [ShimTile], buffer_depth, ofifo_memRef_ty, [], []
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "scale.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out = acquire(
                            ObjectFifoPort.Produce, "out", 1, memRef_ty
                        ).acquired_elem()
                        elem_in = acquire(
                            ObjectFifoPort.Consume, "in", 1, memRef_ty
                        ).acquired_elem()
                        Call(scale_int32, [elem_in, elem_out])
                        objectfifo_release(ObjectFifoPort.Consume, "in", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out", 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
