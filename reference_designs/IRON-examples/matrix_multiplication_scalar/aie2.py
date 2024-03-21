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

# Size of the matrices
M = 4
N = 4
K = 4

A_SIZE = M * K
B_SIZE = K * N
C_SIZE = M * N

objfifo_capacity = 4


def my_matrix_multiplication_scalar():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.xcvc1902)
        def device_body():
            # memRef_ty = T.memref(A_SIZE, T.i32())
            memRef_ty = T.memref(M, N, T.i32())

            # Tile declarations
            ShimTile = tile(6, 0)
            ComputeTile2 = tile(6, 2)

            # AIE-array data movement with object fifos
            # Input
            of_in0 = object_fifo(
                "in0", ShimTile, ComputeTile2, objfifo_capacity, memRef_ty
            )
            of_in1 = object_fifo(
                "in1", ShimTile, ComputeTile2, objfifo_capacity, memRef_ty
            )

            # Output
            of_out0 = object_fifo(
                "out0", ComputeTile2, ShimTile, objfifo_capacity, memRef_ty
            )

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                # Effective while(1)
                for _ in for_(8):
                    elem_in0 = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out0.acquire(ObjectFifoPort.Produce, 1)
                    for n in for_(N):
                        for m in for_(M):
                            for k in for_(K):
                                v0 = memref.load(elem_in0, [m, k])
                                v1 = memref.load(elem_in1, [k, n])
                                v2 = memref.load(elem_out, [m, n])
                                v3 = arith.muli(v0, v1)
                                v4 = arith.addi(v2, v3)
                                memref.store(v4, elem_out, [m, n])
                                yield_([])  # K

                            yield_([])  # N
                        yield_([])  # M

                    of_in0.release(ObjectFifoPort.Consume, 1)
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out0.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement

            tensor_ty = T.memref(A_SIZE, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(inTensorA, inTensorB, outTensor):
                # ipu_dma_memcpy_nd(
                #    metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, C_SIZE]
                # )
                # ipu_dma_memcpy_nd(
                #    metadata="in0", bd_id=1, mem=inTensorA, sizes=[1, 1, 1, A_SIZE]
                # )
                # ipu_dma_memcpy_nd(
                #    metadata="in1", bd_id=1, mem=inTensorB, sizes=[1, 1, 1, B_SIZE]
                # )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_matrix_multiplication_scalar()
