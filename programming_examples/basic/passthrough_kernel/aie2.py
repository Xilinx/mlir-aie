#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils

N = 1024

if len(sys.argv) == 2:
    N = int(sys.argv[1])

lineWidthInBytes = N // 4  # chop input in 4 sub-tensors
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def passthroughKernel():

    @device(AIEDevice.npu)
    def device_body():
        # define types
        memRef_ty = T.memref(lineWidthInBytes, T.ui8())

        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[memRef_ty, memRef_ty, T.i32()]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        if enableTrace:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)
                yield_([])

        #    print(ctx.module.operation.verify())

        tensorSize = N
        tensorSizeInInt32s = tensorSize // 4
        tensor_ty = T.memref(lineWidthInInt32s, T.i32())

        compute_tile2_col, compute_tile2_row = 0, 2

        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, outTensor, notUsed):
            if enableTrace:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    channel=1,
                    bd_id=13,
                    ddr_id=1,
                    size=traceSizeInBytes,
                    offset=tensorSize,
                    start=0x1,
                    stop=0x0,
                    events=[0x4B, 0x22, 0x21, 0x25, 0x2D, 0x2C, 0x1A, 0x4F],
                )

            npu_dma_memcpy_nd(
                metadata="in",
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_dma_memcpy_nd(
                metadata="out",
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    passthroughKernel()
    print(ctx.module)
