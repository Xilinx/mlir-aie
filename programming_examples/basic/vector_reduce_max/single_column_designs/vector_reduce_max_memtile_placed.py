# single_column_designs/vector_reduce_max_memtile_placed.py -*- Python -*-
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

dtype_map = {"i32": np.int32, "bf16": bfloat16}


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
        int_ty = np.ndarray[(O * n_cores,), np.dtype[dtype]]

        # AIE Core Function declarations

        suffix = "_bfloat16" if dtype_str == "bf16" else ""
        reduce_max_vector = external_func(
            f"reduce_max_vector{suffix}", inputs=[op_ty, out_ty, np.int32]
        )
        reduce_max_scalar = external_func(
            f"reduce_max_scalar{suffix}", inputs=[int_ty, out_ty, np.int32]
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
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, mem_ty)

        for i in range(n_cores):
            in_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, op_ty)
            )
            out_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, out_ty)
            )

        if n_cores > 1:
            of_a_offsets = [
                (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
            of_c_offsets = [(O * i) for i in range(n_cores)]
        else:
            of_a_offsets = [0]
            of_c_offsets = [0]

        object_fifo_link(inA, in_fifos, [], of_a_offsets)

        """
        Note: Since DMA BD length needs to be 4 bytes, the stride is used to 
        correctly access data when the datatype size is less than 4 bytes.
        """
        outC = object_fifo(
            "outC",
            MemTile,
            cores[0],
            buffer_depth,
            int_ty,
            dimensionsToStream=[(1, O), (1, 1)],
        )
        object_fifo_link(out_fifos, outC, of_c_offsets, [])

        # Set up a packet-switched flow from core to shim for tracing information
        if n_cores > 1:
            tiles_to_trace = [cores[1]]
        else:
            tiles_to_trace = [cores[0]]

        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # AIE-array data movement with object fifos
        of_out = object_fifo("out", cores[0], ShimTile, buffer_depth, out_ty)

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
                if i == 0:
                    elem_out1 = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in1 = outC.acquire(ObjectFifoPort.Consume, 1)
                    reduce_max_scalar(elem_in1, elem_out1, n_cores)
                    outC.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            if n_cores > 1 and trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(inA, A, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, O], issue_token=True
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
    dev = AIEDevice.npu1_2col
elif opts.device == "npu2":
    dev = AIEDevice.npu2_2col
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
