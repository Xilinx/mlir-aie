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
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils


def vector_softmax(dev, trace_size):

    word_size_in = 2
    N = 262144  # *1024
    M = N // 4
    N_in_bytes = N * word_size_in

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 16
    n_cores_per_col = 4
    n_col = n_cores // n_cores_per_col
    N_per_memtile = n * n_cores_per_col
    tiles = N_div_n // n_cores
    buffer_depth = 2

    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the tile memory
        A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        C_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the memory tile which aggregates across the 4 cores
        A_memTile_ty = np.ndarray[(N_per_memtile,), np.dtype[bfloat16]]
        C_memTile_ty = np.ndarray[(N_per_memtile,), np.dtype[bfloat16]]

        # AIE Core Function declarations

        softmax_bf16_vector = external_func(
            "softmax_bf16", inputs=[tile_ty, tile_ty, np.int32]
        )

        # Tile declarations

        ShimTiles = [tile(i, 0) for i in range(n_col)]
        MemTiles = [tile(i, 1) for i in range(n_col)]

        cores = []

        for i in range(n_col):
            cores.extend([tile(i, 2 + j) for j in range(n_cores_per_col)])

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A and Output C
        of_in0 = object_fifo(
            "of_in0", ShimTiles[0], MemTiles[0], buffer_depth, A_memTile_ty
        )
        of_in1 = object_fifo(
            "of_in1", ShimTiles[1], MemTiles[1], buffer_depth, A_memTile_ty
        )
        of_in2 = object_fifo(
            "of_in2", ShimTiles[2], MemTiles[2], buffer_depth, A_memTile_ty
        )
        of_in3 = object_fifo(
            "of_in3", ShimTiles[3], MemTiles[3], buffer_depth, A_memTile_ty
        )

        of_out0 = object_fifo(
            "of_out0", MemTiles[0], ShimTiles[0], buffer_depth, C_memTile_ty
        )
        of_out1 = object_fifo(
            "of_out1", MemTiles[1], ShimTiles[1], buffer_depth, C_memTile_ty
        )
        of_out2 = object_fifo(
            "of_out2", MemTiles[2], ShimTiles[2], buffer_depth, C_memTile_ty
        )
        of_out3 = object_fifo(
            "of_out3", MemTiles[3], ShimTiles[3], buffer_depth, C_memTile_ty
        )

        for i in range(n_col):
            for j in range(n_cores_per_col):
                inA_fifos.append(
                    object_fifo(
                        f"memA{(i*n_col)+j}",
                        MemTiles[i],
                        cores[(i * n_col) + j],
                        buffer_depth,
                        A_ty,
                    )
                )
                outC_fifos.append(
                    object_fifo(
                        f"memC{(i*n_col)+j}",
                        cores[(i * n_col) + j],
                        MemTiles[i],
                        buffer_depth,
                        C_ty,
                    )
                )

        if n_cores > 1:
            of_a_offsets = [
                (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores_per_col)
                * i
                for i in range(n_cores_per_col)
            ]
            of_c_offsets = [
                (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores_per_col)
                * i
                for i in range(n_cores_per_col)
            ]
        else:
            of_a_offsets = []
            of_c_offsets = []

        inA_fifos_split = [[], [], [], []]
        for i in range(n_col):
            inA_fifos_split[i] = inA_fifos[
                i * n_cores_per_col : (i + 1) * n_cores_per_col
            ]

        object_fifo_link(of_in0, inA_fifos_split[0], [], of_a_offsets)
        object_fifo_link(of_in1, inA_fifos_split[1], [], of_a_offsets)
        object_fifo_link(of_in2, inA_fifos_split[2], [], of_a_offsets)
        object_fifo_link(of_in3, inA_fifos_split[3], [], of_a_offsets)

        outC_fifos_split = [[], [], [], []]
        for i in range(n_col):
            outC_fifos_split[i] = outC_fifos[
                i * n_cores_per_col : (i + 1) * n_cores_per_col
            ]

        object_fifo_link(outC_fifos_split[0], of_out0, of_c_offsets, [])
        object_fifo_link(outC_fifos_split[1], of_out1, of_c_offsets, [])
        object_fifo_link(outC_fifos_split[2], of_out2, of_c_offsets, [])
        object_fifo_link(outC_fifos_split[3], of_out3, of_c_offsets, [])

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "kernels.a")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles):
                        elem_in = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)

                        softmax_bf16_vector(elem_in, elem_out, n)

                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):

            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                    trace_offset=N_in_bytes,
                    ddr_id=1,
                )

            in_task = shim_dma_single_bd_task(
                of_in0, A, offset=0, sizes=[1, 1, 1, M], issue_token=True
            )
            in_task1 = shim_dma_single_bd_task(
                of_in1, A, offset=M, sizes=[1, 1, 1, M], issue_token=True
            )
            in_task2 = shim_dma_single_bd_task(
                of_in2, A, offset=M * 2, sizes=[1, 1, 1, M], issue_token=True
            )
            in_task3 = shim_dma_single_bd_task(
                of_in3, A, offset=M * 3, sizes=[1, 1, 1, M], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out0, C, offset=0, sizes=[1, 1, 1, M], issue_token=True
            )
            out_task1 = shim_dma_single_bd_task(
                of_out1, C, offset=M, sizes=[1, 1, 1, M], issue_token=True
            )
            out_task2 = shim_dma_single_bd_task(
                of_out2, C, offset=M * 2, sizes=[1, 1, 1, M], issue_token=True
            )
            out_task3 = shim_dma_single_bd_task(
                of_out3, C, offset=M * 3, sizes=[1, 1, 1, M], issue_token=True
            )
            dma_start_task(
                in_task,
                in_task1,
                in_task2,
                in_task3,
                out_task,
                out_task1,
                out_task2,
                out_task3,
            )
            dma_await_task(
                in_task,
                in_task1,
                in_task2,
                in_task3,
                out_task,
                out_task1,
                out_task2,
                out_task3,
            )

            trace_utils.gen_trace_done_aie2(ShimTiles[0])


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_4col
    elif device_name == "npu2":
        dev = AIEDevice.npu2_4col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument is not an integer")

with mlir_mod_ctx() as ctx:
    vector_softmax(dev, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
