#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s --check-prefix TRACE
#
# TRACE: aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 65536 : ui32}
# TRACE: aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
# TRACE: aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260527909 : ui32}
# TRACE: aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 757865039 : ui32}
# TRACE: aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
# TRACE: aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
# TRACE: aiex.npu.writebd_shimtile {bd_id = 3 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 2 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
# TRACE: aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 3 : ui32}

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.utils.trace import *

N = 1024

if len(sys.argv) == 2:
    N = int(sys.argv[1])

lineWidthInBytes = N // 4  # chop input in 4 sub-tensors
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = True
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def passthroughKernel():
    with mlir_mod_ctx() as ctx:

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

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(inTensor, outTensor, notUsed):
                if enableTrace:
                    configure_simple_tracing_aie2(
                        ComputeTile2,
                        ShimTile,
                        channel=1,
                        bd_id=3,
                        ddr_id=2,
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

    print(ctx.module)


passthroughKernel()
