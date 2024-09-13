# dma_transpose/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.api.dataflow.inout.simplefifoinout import SimpleFifoInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.dataflow.objectfifolink import MyObjectFifoLink
from aie.api.phys.device import NPU1Col1
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

N = 4096
M = 64
K = 64

if len(sys.argv) == 3:
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = M * K

# TODO: clean up types
memref_ty = ((M, K), np.uint8)

# TODO: rely on depth inference
of_in = MyObjectFifo(2, memref_type=memref_ty)
of_out = MyObjectFifo(2, memref_type=memref_ty)


def core_fn():
    pass


# TODO: clean up placement
worker_program = MyWorker(core_fn, [], coords=(0, 2))
my_link = MyObjectFifoLink([of_in.second], [of_out.first], coords=(0, 2))

inout_program = SimpleFifoInOutProgram(
    of_in.first,
    N,
    of_out.second,
    N,
    # in_sizes=[1, K, M, 1],
    # in_strides=[1, 1, K, 1],
)

my_program = MyProgram(
    NPU1Col1(),
    worker_programs=[worker_program],
    links=[my_link],
    inout_program=inout_program,
)
my_program.resolve_program()

"""
def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            memRef_ty = T.memref(M, K, T.i32())

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in for_(sys.maxsize):
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                # The strides below are configured to read across all rows in the same column
                # Stride of K in dim/wrap 2 skips an entire row to read a full column
                npu_dma_memcpy_nd(
                    metadata="in",
                    bd_id=1,
                    mem=A,
                    sizes=[1, K, M, 1],
                    strides=[1, 1, K, 1],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_passthrough()
"""
