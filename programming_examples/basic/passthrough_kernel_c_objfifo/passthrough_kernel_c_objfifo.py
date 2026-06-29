# passthrough_kernel_c_objfifo/passthrough_kernel_c_objfifo.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# Variant of passthrough_kernel that demonstrates the C ObjectFIFO API.
# Instead of compiler-managed acquire/release, this design passes lock IDs
# and buffer references to the C kernel via aie.objectfifo.lock and
# aie.objectfifo.buffer, letting the kernel call acquire/release directly
# using aie_objectfifo.h.

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def my_passthrough_kernel(dev, in1_size, out_size):
    in1_dtype = np.uint8
    out_dtype = np.uint8

    N = in1_size // in1_dtype(0).nbytes
    lineWidthInBytes = N // 4  # chop input in 4 sub-tensors

    assert (
        out_size == in1_size
    ), "Output buffer size must be equal to input buffer size."

    @device(dev)
    def device_body():
        # define types
        vector_ty = np.ndarray[(N,), np.dtype[in1_dtype]]
        line_ty = np.ndarray[(lineWidthInBytes,), np.dtype[in1_dtype]]

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine",
            inputs=[
                line_ty,  # in buffer 0
                line_ty,  # in buffer 1
                line_ty,  # out buffer 0
                line_ty,  # out buffer 1
                T.index(),  # in acq_lock
                T.index(),  # in rel_lock
                T.index(),  # out acq_lock
                T.index(),  # out rel_lock
            ],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, line_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, line_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "kernel.o")
        def core_body():
            # Pass both ping-pong buffers and lock IDs to C kernel
            in_buf0 = of_in.get_buffer(0)
            in_buf1 = of_in.get_buffer(1)
            in_acq, in_rel = of_in.get_lock(ObjectFifoPort.Consume)

            out_buf0 = of_out.get_buffer(0)
            out_buf1 = of_out.get_buffer(1)
            out_acq, out_rel = of_out.get_lock(ObjectFifoPort.Produce)

            # C kernel owns the compute loop and buffer rotation
            passThroughLine(
                in_buf0,
                in_buf1,
                out_buf0,
                out_buf1,
                in_acq,
                in_rel,
                out_acq,
                out_rel,
            )

        @runtime_sequence(vector_ty, vector_ty, vector_ty)
        def sequence(inTensor, outTensor, notUsed):
            in_task = shim_dma_single_bd_task(
                of_in, inTensor, sizes=[1, 1, 1, N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out, outTensor, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
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

with mlir_mod_ctx() as ctx:
    my_passthrough_kernel(dev, in1_size, out_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
