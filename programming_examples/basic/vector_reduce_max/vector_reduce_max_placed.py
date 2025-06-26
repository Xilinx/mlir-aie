# vector_reduce_max/vector_reduce_max_cascade.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape
from ml_dtypes import bfloat16

import aie.utils.trace as trace_utils

dtype_map = {
    "bf16": bfloat16,
    "i32": np.int32,
}


def my_reduce_max(dev, in1_size, out_size, dtype_str, trace_size):
    n_cores = 8
    in_dtype = dtype_map[dtype_str]
    out_dtype = dtype_map[dtype_str]

    N = in1_size // in_dtype(0).nbytes
    O = out_size // out_dtype(0).nbytes

    n_mem_elems = 128
    elems_per_core = n_mem_elems

    n_channels = n_cores
    N_per_channel = N // n_channels

    num_iter = N // (n_mem_elems * n_channels)

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    buffer_depth = 2

    if n_cores > 8:
        raise ValueError("This design does not support more than 8 cores.")

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(N,), np.dtype[in_dtype]]
        mem_ty = np.ndarray[(n_mem_elems,), np.dtype[in_dtype]]
        op_ty = np.ndarray[(elems_per_core,), np.dtype[in_dtype]]
        out_ty = np.ndarray[(O,), np.dtype[out_dtype]]

        # AIE Core Function declarations
        if dtype_str == "bf16":
            reduce_max_vector = external_func(
                "reduce_max_vector_bfloat16", [op_ty, out_ty, np.int32]
            )
            compute_max = external_func(
                "compute_max_bfloat16", [out_ty, out_ty, out_ty]
            )
        else:
            reduce_max_vector = external_func(
                "reduce_max_vector", [op_ty, out_ty, np.int32]
            )
            compute_max = external_func("compute_max", [out_ty, out_ty, out_ty])

        # Tile declarations
        # Create an array of ShimTiles, one for every two cores
        shimtiles = [tile(i, 0) for i in range((n_cores + 1) // 2)]
        # Distribute cores: first 4 in col 0, next in col 1
        cores = [tile(i // 4, 2 + (i % 4)) for i in range(n_cores)]

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A and Output C

        for i in range(n_cores):
            # For every 2 cores, use the next shimtile
            shimtile_idx = i // 2
            inA_fifos.append(
                object_fifo(
                    f"memA{i}", shimtiles[shimtile_idx], cores[i], buffer_depth, op_ty
                )
            )

        # Output FIFO for core 0: connect to core 1 if n_cores > 1, else to shimtile 0, others to core 1 or core 6 as needed
        outC_fifos.append(
            object_fifo(
                f"memC0",
                cores[0],
                cores[1] if n_cores > 1 else shimtiles[0],
                buffer_depth,
                out_ty,
            )
        )
        if n_cores > 1:
            outC_fifos.append(
                object_fifo(
                    f"memC1",
                    cores[1],
                    (
                        cores[4]
                        if n_cores == 5
                        else (cores[5] if n_cores > 5 else shimtiles[0])
                    ),
                    buffer_depth,
                    out_ty,
                )
            )
        for i in range(2, n_cores):
            # For n_cores > 5, cores[6] connects to core 1, others to core 6
            if n_cores > 5 and i == 5:
                outC_fifos.append(
                    object_fifo(
                        f"memC{i}", cores[5], shimtiles[1], buffer_depth, out_ty
                    )
                )
            elif n_cores > 5 and i == 4:
                outC_fifos.append(
                    object_fifo(f"memC{i}", cores[4], cores[5], buffer_depth, out_ty)
                )
            elif i >= 6:
                outC_fifos.append(
                    object_fifo(f"memC{i}", cores[i], cores[5], buffer_depth, out_ty)
                )
            elif n_cores == 5 and i == 4:
                outC_fifos.append(
                    object_fifo(
                        f"memC{i}", cores[i], shimtiles[1], buffer_depth, out_ty
                    )
                )
            else:
                outC_fifos.append(
                    object_fifo(f"memC{i}", cores[i], cores[1], buffer_depth, out_ty)
                )

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0] if n_cores == 1 else cores[5]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, shimtiles[0])

        if dtype_str == "bf16":
            min_val = np.array([bfloat16(float(-4.0))], dtype=bfloat16)
        else:
            min_val = np.array([np.iinfo(np.int32).min], dtype=np.int32)

        # Set up compute tiles
        for i in range(n_cores):
            nextC_buffer = buffer(
                tile=cores[i],
                datatype=np.ndarray[(O,), np.dtype[out_dtype]],
                name=f"elem_out_{i}",
                initial_value=min_val,
            )
            tmp_buffer = buffer(
                tile=cores[i],
                datatype=np.ndarray[(O,), np.dtype[out_dtype]],
                name=f"tmp_buffer_{i}",
                initial_value=min_val,
            )

            @core(cores[i], "reduce_max.cc.o")
            def core_body():
                elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                for _ in range_(num_iter):
                    elem_in = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                    reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
                    compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
                    inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                if i == 4 and n_cores == 5:
                    # Final reduction here
                    partial_result = outC_fifos[1].acquire(ObjectFifoPort.Consume, 1)
                    compute_max(partial_result, nextC_buffer, elem_out)
                    outC_fifos[1].release(ObjectFifoPort.Consume, 1)
                elif i != 1 and i != 5:
                    elem_out[0] = nextC_buffer[0]

                if i == 5:
                    # Final in core6
                    cores_per_col = n_cores - 4
                    elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    inputs = []
                    start_j = 4 if i == 5 else 0
                    inputs = [
                        outC_fifos[j].acquire(ObjectFifoPort.Consume, 1)
                        for j in range(start_j, 4 + cores_per_col)
                        if j != i
                    ]
                    inputs.append(outC_fifos[1].acquire(ObjectFifoPort.Consume, 1))

                    for idx in range(cores_per_col):
                        if idx < cores_per_col - 1:
                            compute_max(
                                inputs[idx],
                                nextC_buffer,
                                nextC_buffer,
                            )
                        else:
                            compute_max(inputs[idx], nextC_buffer, elem_out)

                    for j, input_elem in enumerate(inputs):
                        outC_fifos[4 + j if j != 1 else j].release(
                            ObjectFifoPort.Consume, 1
                        )
                if i == 1:
                    cores_per_col = min(4, n_cores)
                    inputs = []
                    start_j = 5 if i == 6 else 0
                    inputs = [
                        outC_fifos[j].acquire(ObjectFifoPort.Consume, 1)
                        for j in range(start_j, cores_per_col)
                        if j != i
                    ]

                    for idx in range(cores_per_col - 1):
                        if idx < cores_per_col - 2:
                            compute_max(
                                inputs[idx],
                                nextC_buffer,
                                nextC_buffer,
                            )
                        else:
                            compute_max(inputs[idx], nextC_buffer, elem_out)

                    for j, input_elem in enumerate(inputs):
                        outC_fifos[j if j < i else j + 1].release(
                            ObjectFifoPort.Consume, 1
                        )
                outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=shimtiles[0],
                    trace_size=trace_size,
                )
            in_task = []
            for i in range(n_channels):
                in_task.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i],
                        A,
                        offset=N_per_channel * i,
                        sizes=[1, 1, 1, N_per_channel],
                    )
                )
            out_task = shim_dma_single_bd_task(
                outC_fifos[
                    (
                        0
                        if n_cores == 1
                        else 1 if n_cores < 5 else 4 if n_cores == 5 else 5
                    )
                ],
                C,
                sizes=[1, 1, 1, O],
                issue_token=True,
            )
            for i in range(n_cores):
                dma_start_task(in_task[i])
            dma_start_task(out_task)
            dma_await_task(out_task)

            trace_utils.gen_trace_done_aie2(shimtiles[0])


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")

p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument("-dt", "--dtype", required=True, dest="dtype", help="Datatype")
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
    dev = AIEDevice.npu1
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
dtype = str(opts.dtype)
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_reduce_max(dev, in1_size, out_size, dtype, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
