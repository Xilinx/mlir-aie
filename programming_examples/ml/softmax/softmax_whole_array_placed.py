#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys
import argparse

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils


def vector_softmax(dev, trace_size, n_col, n_cores_per_col, N):

    word_size_in = 2
    # N = 262144  # *1024
    N_in_bytes = N * word_size_in

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = n_col * n_cores_per_col
    N_per_shimtile = N // n_col
    N_per_memtile = n * n_cores_per_col
    tiles = N_div_n // n_cores
    buffer_depth = 2

    if dev == AIEDevice.npu1 and n_col > 4:
        raise ValueError(
            "[ERROR] NPU1 device only supports 4 columns. Please set n_col <= 4"
        )
    if dev == AIEDevice.npu2_4col and n_col > 8:
        raise ValueError(
            "[ERROR] NPU2 device only supports 8 columns. Please set n_col <= 8"
        )
    if n_cores_per_col > 4:
        raise ValueError(
            "[ERROR] Only 4 cores per column are supported. Please set n_cores_per_col <= 4"
        )

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

        of_in = []
        for i in range(n_col):
            of_in.append(
                object_fifo(
                    f"of_in{i}", ShimTiles[i], MemTiles[i], buffer_depth, A_memTile_ty
                )
            )

        of_out = []
        for i in range(n_col):
            of_out.append(
                object_fifo(
                    f"of_out{i}", MemTiles[i], ShimTiles[i], buffer_depth, C_memTile_ty
                )
            )
        # Create object FIFOs to split input A across cores of a column and
        # join output C across cores of a column.
        for i in range(n_col):
            for j in range(n_cores_per_col):
                # FIFO for input A from memory tile to core
                inA_fifos.append(
                    object_fifo(
                        f"memA{(i*n_cores_per_col)+j}",
                        MemTiles[i],
                        cores[(i * n_cores_per_col) + j],
                        buffer_depth,
                        A_ty,
                    )
                )
                # FIFO for output C from core to memory tile
                outC_fifos.append(
                    object_fifo(
                        f"memC{(i*n_cores_per_col)+j}",
                        cores[(i * n_cores_per_col) + j],
                        MemTiles[i],
                        buffer_depth,
                        C_ty,
                    )
                )

        # Offsets for splitting input A and joining output C across cores
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

        # Split the input FIFOs for each column into groups corresponding to the cores
        # in that column and link them to the memory tile FIFOs.
        inA_fifos_split = [[] for _ in range(n_col)]
        for i in range(n_col):
            inA_fifos_split[i] = inA_fifos[
                i * n_cores_per_col : (i + 1) * n_cores_per_col
            ]

        # Link the input FIFOs from memory tiles to the cores in each column
        for i in range(n_col):
            object_fifo_link(of_in[i], inA_fifos_split[i], [], of_a_offsets)

        # Split the output FIFOs for each column into groups corresponding to the cores
        # in that column and link them to the memory tile FIFOs.
        outC_fifos_split = [[] for _ in range(n_col)]
        for i in range(n_col):
            outC_fifos_split[i] = outC_fifos[
                i * n_cores_per_col : (i + 1) * n_cores_per_col
            ]

        # Link the output FIFOs from the cores to the memory tiles in each column
        for i in range(n_col):
            object_fifo_link(outC_fifos_split[i], of_out[i], of_c_offsets, [])

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTiles[i])

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
                    shim=ShimTiles[0],
                    trace_size=trace_size,
                    trace_offset=N_in_bytes,
                    ddr_id=1,
                )

            in_tasks = []
            out_tasks = []

            # Loop through each column to set up DMA tasks for input and output
            for i in range(n_col):
                # Distributing host buffer (A) to the shim tiles
                in_tasks.append(
                    shim_dma_single_bd_task(
                        of_in[i],
                        A,
                        offset=N_per_shimtile * i,
                        sizes=[1, 1, 1, N_per_shimtile],
                    )
                )
                # Joining output from the shim tiles and writing to host buffer (C)
                out_tasks.append(
                    shim_dma_single_bd_task(
                        of_out[i],
                        C,
                        offset=N_per_shimtile * i,
                        sizes=[1, 1, 1, N_per_shimtile],
                        issue_token=True,
                    )
                )

            dma_start_task(*in_tasks, *out_tasks)
            dma_await_task(*out_tasks)

            trace_utils.gen_trace_done_aie2(cores[0])


def main():
    parser = argparse.ArgumentParser(prog="softmax_whole_array_placed")
    parser.add_argument(
        "device_name",
        choices=["npu", "npu2"],
        default="npu",
        help="Device name (npu or npu2)",
    )
    parser.add_argument(
        "trace_size_pos",
        nargs="?",
        type=int,
        default=0,
        help="Trace size (optional positional, default: 0)",
    )
    parser.add_argument(
        "--trace_size",
        dest="trace_size_flag",
        type=int,
        default=0,
        help="Trace size (optional flag, default: 0)",
    )
    parser.add_argument(
        "--n_col",
        type=int,
        default=4,
        help="Number of columns (default: 4)",
    )
    parser.add_argument(
        "--n_cores_per_col",
        type=int,
        default=4,
        help="Number of cores per column (default: 4)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=262144,
        help="Size of the input vector (default: 262144)",
    )

    args = parser.parse_args()

    trace_size = args.trace_size_flag if args.trace_size_flag != 0 else args.trace_size_pos

    if args.device_name == "npu":
        dev = AIEDevice.npu1
    elif args.device_name == "npu2":
        dev = AIEDevice.npu2_4col
    else:
        # This should be caught by argparse choices, but just in case
        raise ValueError(f"[ERROR] Device name {args.device_name} is unknown")

    with mlir_mod_ctx() as ctx:
        vector_softmax(dev, trace_size, args.n_col, args.n_cores_per_col, args.size)
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)


if __name__ == "__main__":
    main()
