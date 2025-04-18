#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape


import aie.utils.trace as trace_utils


def my_eltwise_add(dev, in1_size, in2_size, out_size, trace_size):
    in1_dtype = bfloat16
    in2_dtype = bfloat16
    out_dtype = bfloat16

    tensor_size = in1_size // in1_dtype(0).nbytes

    # Tile sizes
    tile_size = 1024
    tensor_div_tile = tensor_size // tile_size

    n_cores = 2
    tiles = tensor_div_tile // n_cores
    buffer_depth = 2

    assert in2_size == in1_size, "input2 buffer size must match input1 buffer size."
    assert out_size == in1_size, "Output buffer size must match input1 buffer size."

    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the tile memory
        A_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        B_ty = np.ndarray[(tile_size,), np.dtype[in2_dtype]]
        C_ty = np.ndarray[(tile_size,), np.dtype[out_dtype]]

        # Type used in the memory tile which aggregates across the 2 cores
        A_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in1_dtype]]
        B_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[in2_dtype]]
        C_memTile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[out_dtype]]

        # AIE Core Function declarations

        eltwise_add_bf16_scalar = external_func(
            "eltwise_add_bf16_scalar", inputs=[tile_ty, tile_ty, tile_ty]
        )
        eltwise_add_bf16_vector = external_func(
            "eltwise_add_bf16_vector", inputs=[tile_ty, tile_ty, tile_ty]
        )

        # Tile declarations
        ShimTile = tile(0, 0)

        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        inB_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, A_memTile_ty)
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, A_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inA, inA_fifos, [], of_offsets)

        # Input B
        inB = object_fifo("inB", ShimTile, MemTile, buffer_depth, B_memTile_ty)
        for i in range(n_cores):
            inB_fifos.append(
                object_fifo(f"memB{i}", MemTile, cores[i], buffer_depth, B_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(B_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inB, inB_fifos, [], of_offsets)

        # Output C
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, C_ty)
            )
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, C_memTile_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifos, outC, of_offsets, [])

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "add.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(tiles):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        eltwise_add_bf16_vector(elem_in_a, elem_in_b, elem_out)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0], cores[1]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # To/from AIE-array data movement
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[out_dtype]]

        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):

            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            a_task = shim_dma_single_bd_task(
                inA,
                A,
                sizes=[1, 1, 1, tensor_size],
                issue_token=True,
            )
            b_task = shim_dma_single_bd_task(
                inB,
                B,
                sizes=[1, 1, 1, tensor_size],
                issue_token=True,
            )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(a_task, b_task, c_task)
            dma_await_task(a_task, b_task, c_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 5:
    raise ValueError(
        "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
    )


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(opts.in1_size)
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_eltwise_add(dev, in1_size, in2_size, out_size, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
