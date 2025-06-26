# single_core_designs/vector_reduce_max_placed.py -*- Python -*-
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

import aie.utils.trace as trace_utils

from ml_dtypes import bfloat16

dtype_map = {
    "bf16": bfloat16,
    "i32": np.int32,
}


def my_reduce_max(dev, in1_size, out_size, dtype_str, trace_size):
    in1_dtype = dtype_map[dtype_str]
    out_dtype = dtype_map[dtype_str]

    N = in1_size // in1_dtype(0).nbytes
    O = out_size // out_dtype(0).nbytes

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    buffer_depth = 2

    @device(dev)
    def device_body():
        in_ty = np.ndarray[(N,), np.dtype[in1_dtype]]
        out_ty = np.ndarray[(O,), np.dtype[out_dtype]]

        # AIE Core Function declarations
        if dtype_str == "bf16":
            reduce_max_vector = external_func(
                "reduce_max_vector_bfloat16", inputs=[in_ty, out_ty, np.int32]
            )
        else:
            reduce_max_vector = external_func(
                "reduce_max_vector", inputs=[in_ty, out_ty, np.int32]
            )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, in_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, out_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "reduce_max.cc.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                reduce_max_vector(elem_in, elem_out, N)
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

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
