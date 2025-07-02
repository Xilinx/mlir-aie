# single_column_designs/vector_reduce_max_shared_placed.py -*- Python -*-
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
    n_cores = 4
    dtype = dtype_map[dtype_str]

    N = in1_size // dtype(0).nbytes
    O = out_size // dtype(0).nbytes

    n_mem_elems = 2048
    elems_per_core = n_mem_elems // n_cores
    num_iter = N // n_mem_elems

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    buffer_depth = 2

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(N,), np.dtype[dtype]]
        mem_ty = np.ndarray[(n_mem_elems,), np.dtype[dtype]]
        op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
        out_ty = np.ndarray[(O,), np.dtype[dtype]]

        # AIE Core Function declarations
        suffix = "_bfloat16" if dtype_str == "bf16" else ""
        reduce_max_vector = external_func(
            f"reduce_max_vector{suffix}", inputs=[op_ty, out_ty, np.int32]
        )
        compute_max = external_func(
            f"compute_max{suffix}", inputs=[out_ty, out_ty, out_ty]
        )
        min_val = (
            np.array([bfloat16(float("-inf"))], dtype=dtype)
            if dtype_str == "bf16"
            else np.array([np.iinfo(dtype).min], dtype=dtype)
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)

        cores = [tile(0, 2 + i) for i in range(n_cores)]

        in_fifos = []
        out_fifos = []

        # AIE-array data movement with object fifos
        # Input A and Output C
        of_in = object_fifo("of_in", ShimTile, MemTile, buffer_depth, mem_ty)

        for i in range(n_cores):
            in_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, op_ty)
            )
        out_fifos.append(
            object_fifo(
                f"memC{0}",
                cores[0],
                ShimTile if n_cores == 1 else cores[1],
                buffer_depth,
                out_ty,
            )
        )
        if n_cores > 1:
            out_fifos.append(
                object_fifo(f"memC{1}", cores[1], ShimTile, buffer_depth, out_ty)
            )
        for i in range(2, n_cores):
            out_fifos.append(
                object_fifo(f"memC{i}", cores[i], cores[1], buffer_depth, out_ty)
            )

        if n_cores > 1:
            of_a_offsets = [
                (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_a_offsets = [0]

        object_fifo_link(of_in, in_fifos, [], of_a_offsets)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # Set up compute tiles
        for i in range(n_cores):
            nextC_buffer = buffer(
                tile=cores[i],
                datatype=np.ndarray[(O,), np.dtype[dtype]],
                name=f"elem_out_{i}",
                initial_value=min_val,
            )
            tmp_buffer = buffer(
                tile=cores[i],
                datatype=np.ndarray[(O,), np.dtype[dtype]],
                name=f"tmp_buffer_{i}",
                initial_value=min_val,
            )
            if i != 1:

                @core(cores[i], "reduce_max.cc.o")
                def core_body():
                    elem_out = out_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    for _ in range_(num_iter):
                        elem_in = in_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
                        compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
                        in_fifos[i].release(ObjectFifoPort.Consume, 1)
                    elem_out[0] = nextC_buffer[0]
                    out_fifos[i].release(ObjectFifoPort.Produce, 1)

            else:

                @core(cores[i], "reduce_max.cc.o")
                def core_body():
                    for _ in range_(num_iter):
                        elem_in = in_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
                        compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
                        in_fifos[i].release(ObjectFifoPort.Consume, 1)

                    elem_out = out_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    inputs = []
                    for j in range(n_cores):
                        if j != i:
                            inputs.append(
                                out_fifos[j].acquire(ObjectFifoPort.Consume, 1)
                            )

                    for idx in range(n_cores - 1):
                        if idx < n_cores - 2:
                            compute_max(
                                inputs[idx],
                                nextC_buffer,
                                nextC_buffer,
                            )
                        else:
                            compute_max(inputs[idx], nextC_buffer, elem_out)

                    for j, input_elem in enumerate(inputs):
                        out_fifos[j if j < i else j + 1].release(
                            ObjectFifoPort.Consume, 1
                        )

                    out_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(of_in, A, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                out_fifos[0 if n_cores == 1 else 1],
                C,
                sizes=[1, 1, 1, O],
                issue_token=True,
            )
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


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
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2_1col
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
