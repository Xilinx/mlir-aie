# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils


def passthroughKernel(vector_size, trace_size):
    N = vector_size
    lineWidthInBytes = N // 4  # chop input in 4 sub-tensors
    
    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need command line arguments (Device name, Vector size)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        # define types
        vector_ty = np.ndarray[(N,), np.dtype[np.uint8]]
        line_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint8]]

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[line_ty, line_ty, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, line_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, line_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "passThrough.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
                passThroughLine(elemIn, elemOut, lineWidthInBytes)
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        #    print(ctx.module.operation.verify())

        @runtime_sequence(vector_ty, vector_ty, vector_ty)
        def sequence(inTensor, outTensor, notUsed):
            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    ddr_id=1,
                    size=trace_size,
                    offset=N,
                )

            npu_dma_memcpy_nd(
                metadata=of_in,
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, N],
                issue_token=True,
            )
            npu_dma_memcpy_nd(
                metadata=of_out,
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, N],
            )
            dma_wait(of_in, of_out)


try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    passthroughKernel(vector_size, trace_size)
    print(ctx.module)
