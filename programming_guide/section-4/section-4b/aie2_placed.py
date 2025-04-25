# section-4/section-4b/aie2.py -*- Python -*-
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


def my_vector_scalar_mul(dev, in1_size, in2_size, out_size, trace_size):
    in1_dtype = np.int32
    in2_dtype = np.int32
    out_dtype = np.int32

    tensor_size = in1_size // in1_dtype(0).nbytes
    num_sub_vectors = 4
    tile_size = tensor_size // num_sub_vectors

    assert in2_size == 4, "2nd input buffer must be size 4 (4 bytes = 1 integer)."
    assert out_size == in1_size, "Output buffer size must match input buffer size."

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
        tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
        scalar_ty = np.ndarray[(1,), np.dtype[in2_dtype]]

        # AIE Core Function declarations
        scale_scalar = external_func(
            "vector_scalar_mul_aie_scalar",
            inputs=[tile_ty, tile_ty, scalar_ty, in2_dtype],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_factor = object_fifo("infactor", ShimTile, ComputeTile2, 2, scalar_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in range_(num_sub_vectors):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale_scalar(elem_in, elem_out, elem_factor, tile_size)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [ComputeTile2, ShimTile]
        if trace_size > 0:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            if trace_size > 0:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    shim=ShimTile,
                    trace_size=trace_size,
                )

            in_task = shim_dma_single_bd_task(
                of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
            )
            in_factor_task = shim_dma_single_bd_task(
                of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, tensor_size], issue_token=True
            )

            dma_start_task(in_task, in_factor_task, out_task)
            dma_await_task(in_task, in_factor_task, out_task)

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
if in1_size % 128 != 0 or in1_size < 1024:
    print(
        "In1 buffer size must be a multiple of 128 (so len is multiple of 64) and greater than or equal to 1024 (so len >= 512)"
    )
    raise ValueError
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

with mlir_mod_ctx() as ctx:
    my_vector_scalar_mul(dev, in1_size, in2_size, out_size, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
