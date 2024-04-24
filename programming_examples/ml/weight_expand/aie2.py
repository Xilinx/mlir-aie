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


def my_expand():

    SF_BLOCK_SIZE = 32
    word_size_in = 2
    sf_word_size_in = 2
    N = 65536

    N_in_bytes = (N // word_size_in) + (N / SF_BLOCK_SIZE) * sf_word_size_in

    A_sz_in_i32s = (N // 8) + (
        N // SF_BLOCK_SIZE
    ) // 2  # They are 4 bits per element, we need to add on the scale factors later though
    B_sz_in_i32s = N // 2  # Returning 16 bits at the moment

    # Tile sizes
    n = 1024
    block_size = 32
    sf_size = n // block_size

    input_buffer_size_bytes = (n // 2) + (
        sf_size * 2
    )  # They are bfloat16 sfs after the private values
    output_buffer_size_bytes = n * 2  # The unscaled values

    N_div_n = N // n

    n_cores = 1
    tiles = N_div_n // n_cores
    buffer_depth = 2

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu)
        def device_body():
            memRef_i_ty = T.memref(
                input_buffer_size_bytes, T.i8()
            )  # Just think of the input as a raw byte buffer
            memRef_o_ty = T.memref(output_buffer_size_bytes, T.i8())  # For now

            # AIE Core Function declarations

            expand_int4_to_bfloat16 = external_func(
                "expand_int4_to_bfloat16", inputs=[memRef_i_ty, memRef_o_ty]
            )

            # Tile declarations
            ShimTile = tile(0, 0)

            MemTile = tile(0, 1)
            core0 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input
            inA = object_fifo("inA", ShimTile, core0, buffer_depth, memRef_i_ty)

            # Output B
            outB = object_fifo("outB", core0, ShimTile, buffer_depth, memRef_o_ty)

            # Set up compute tiles
            @core(core0, "expand.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles):
                        elem_out = outB.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = inA.acquire(ObjectFifoPort.Consume, 1)

                        call(expand_int4_to_bfloat16, [elem_in, elem_out])
                        inA.release(ObjectFifoPort.Consume, 1)
                        outB.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty)
            def sequence(A, C):

                npu_dma_memcpy_nd(
                    metadata="outB", bd_id=0, mem=C, sizes=[1, 1, 1, B_sz_in_i32s]
                )
                npu_dma_memcpy_nd(
                    metadata="inA", bd_id=1, mem=A, sizes=[1, 1, 1, A_sz_in_i32s]
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_expand()
