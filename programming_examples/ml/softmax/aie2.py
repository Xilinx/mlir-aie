#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_
from aie.extras.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils


def vector_softmax(trace_size):

    word_size_in = 2
    N = 262144  # *1024
    N_in_bytes = N * word_size_in

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores
    buffer_depth = 2

    @device(AIEDevice.npu1_1col)
    def device_body():
        tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the tile memory
        A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        C_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the memory tile which aggregates across the 4 cores
        A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
        C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

        # AIE Core Function declarations

        softmax_bf16_vector = external_func(
            "softmax_bf16_vector", inputs=[tile_ty, tile_ty]
        )

        # Tile declarations
        ShimTile = tile(0, 0)

        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A and Output C
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, A_memTile_ty)
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, C_memTile_ty)

        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, A_ty)
            )
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, C_ty)
            )

        if n_cores > 1:
            of_a_offsets = [
                (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
            of_c_offsets = [
                (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_a_offsets = []
            of_c_offsets = []
        object_fifo_link(inA, inA_fifos, [], of_a_offsets)
        object_fifo_link(outC_fifos, outC, of_c_offsets, [])

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(cores[0], WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "kernels.a")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)

                        softmax_bf16_vector(elem_in_a, elem_out)

                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):

            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    cores[0],
                    ShimTile,
                    ddr_id=1,
                    size=trace_size,
                    offset=N_in_bytes,
                )
            npu_dma_memcpy_nd(
                metadata=inA, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
            )
            npu_dma_memcpy_nd(metadata=outC, bd_id=0, mem=C, sizes=[1, 1, 1, N])
            dma_wait(inA, outC)


try:
    trace_size = 0 if (len(sys.argv) != 2) else int(sys.argv[1])
except ValueError:
    print("Argument is not an integer")

with mlir_mod_ctx() as ctx:
    vector_softmax(trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
