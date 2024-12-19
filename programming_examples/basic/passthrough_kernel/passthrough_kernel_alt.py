# passthrough_kernel/passthrough_kernel_alt.py -*- Python -*-
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


def passthroughKernel(dev, vector_size, trace_size):
    N = vector_size
    lineWidthInBytes = N // 4  # chop input in 4 sub-tensors

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
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)
        
        # AIE-array data movement with object fifos
        # Input
        # Origianal code
        # of_in = object    _fifo("in", ShimTile, ComputeTile2, 2, line_ty)
        # of_out = object_fifo("out", ComputeTile2, ShimTile, 2, line_ty)
        
################################################################################################      
        
        # Passes if Object FIFO size is the same
        # of_in0 = object_fifo("in0", ShimTile, MemTile, 2, line_ty)
        # of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, line_ty,
        #                         (
        #                             [
        #                                 (2, 512),
        #                                 (512, 1),
        #                             ]
        #                         ),
        # )
        # object_fifo_link(of_in0, of_in1)
        
################################################################################################

        # Does not complete and times out if Object FIFO size is different and there is transforms
        of_in0 = object_fifo("in0", ShimTile, MemTile, 2, vector_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, line_ty,
                                (
                                    [
                                        # comment to make it work
                                        (8, 512), # 8 to account for different fifo sizes
                                        (512, 1),
                                    ]
                                ),
        )
        object_fifo_link(of_in0, of_in1)
        
        # After Stateful Transform
        # ^bb4:  // 2 preds: ^bb3, ^bb5
        # aie.use_lock(%in0_cons_cons_lock, AcquireGreaterEqual, 1)
        # aie.dma_bd(%in0_cons_buff_0 : memref<4096xui8>, 0, 1024, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        # aie.use_lock(%in0_cons_prod_lock, Release, 1)
        # aie.next_bd ^bb5
        
        # As can be seen with streaming transformation memref and 1024 do not align. Commenting out the transformation alings this 
        # so that "aie.dma_bd(%in0_cons_buff_0 : memref<4096xui8>, 0, 4096)" and functions correctly just for the passthough kernel
        
################################################################################################

        # Output
        of_out0 = object_fifo("out0", MemTile, ShimTile, 2, line_ty)
        of_out1 = object_fifo("out1", ComputeTile2, MemTile, 2, line_ty)
        object_fifo_link(of_out1, of_out0)

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "passThrough.cc.o")
        def core_body():
            for _ in range_(sys.maxsize):
                elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in1.acquire(ObjectFifoPort.Consume, 1)
                passThroughLine(elemIn, elemOut, lineWidthInBytes)
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)

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
            in_task = shim_dma_single_bd_task(
                of_in0, inTensor, sizes=[1, 1, 1, N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out0, outTensor, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_1col
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    vector_size = int(sys.argv[2])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 4) else int(sys.argv[3])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    passthroughKernel(dev, vector_size, trace_size)
    print(ctx.module)
