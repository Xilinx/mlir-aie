# vector_reduce_max/vector_reduce_max_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape

import aie.utils.trace as trace_utils


def my_reduce_max(dev, in1_size, out_size, trace_size, n_cores):
    in_dtype = np.int32
    out_dtype = np.int32

    N = in1_size // in_dtype(0).nbytes
    M = N // n_cores
    O = out_size // out_dtype(0).nbytes

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    buffer_depth = 2

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(N,), np.dtype[in_dtype]]
        inA_ty = np.ndarray[(M,), np.dtype[in_dtype]]
        out_ty = np.ndarray[(O,), np.dtype[out_dtype]]

        # AIE Core Function declarations
        reduce_max_vector = external_func(
            "reduce_max_vector", inputs=[inA_ty, out_ty, np.int32]
        )
        compute_max = external_func("compute_max", inputs=[out_ty, out_ty, out_ty])

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)

        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A and Output C
        of_in = object_fifo("of_in", ShimTile, MemTile, buffer_depth, in_ty)

        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, inA_ty)
            )
            if i == 0:
                outC_fifos.append(
                    object_fifo(f"memC{i}", cores[i], ShimTile, buffer_depth, out_ty)
                )
            else:
                outC_fifos.append(
                    object_fifo(
                        f"memC{i}", cores[i], cores[i - 1], buffer_depth, out_ty
                    )
                )

        if n_cores > 1:
            of_a_offsets = [
                (np.prod(np_ndarray_type_get_shape(in_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_a_offsets = []

        object_fifo_link(of_in, inA_fifos, [], of_a_offsets)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [cores[0]]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # Set up compute tiles
        for i in range(n_cores):
            if i == n_cores - 1:

                @core(cores[i], "reduce_max.cc.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        reduce_max_vector(elem_in, elem_out, M)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

            else:
                nextC_buffer = buffer(
                    tile=cores[i],
                    datatype=np.ndarray[(1,), np.dtype[out_dtype]],
                    name=f"elem_out_{i}",
                )

                @core(cores[i], "reduce_max.cc.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_in = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        reduce_max_vector(elem_in, nextC_buffer, M)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)

                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in = outC_fifos[i + 1].acquire(ObjectFifoPort.Consume, 1)
                        elem_out = compute_max(elem_in, nextC_buffer, elem_out)
                        outC_fifos[i + 1].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                outC_fifos[0], C, sizes=[1, 1, 1, O], issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)

            trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")

p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
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
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_reduce_max(dev, in1_size, out_size, trace_size, n_cores=4)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
