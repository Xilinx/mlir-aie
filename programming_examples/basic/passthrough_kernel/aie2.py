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

N = 1024

if len(sys.argv) == 2:
    N = int(sys.argv[1])

lineWidthInBytes = N // 4  # chop input in 4 sub-tensors
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def passthroughKernel():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
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
                flow(ComputeTile2, "Trace", 0, ShimTile, "DMA", 1)

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
                    # Trace output

                    # Trace_Event0, Trace_Event1: Select which events to trace.
                    # Note that the event buffers only appear to be transferred to DDR in
                    # bursts of 256 bytes. If less than 256 bytes are written, you may not
                    # see trace output, or only see it on the next iteration of your
                    # kernel invocation, as the buffer gets filled up. Note that, even
                    # though events are encoded as 4 byte words, it may take more than 64
                    # events to fill the buffer to 256 bytes and cause a flush, since
                    # multiple repeating events can be 'compressed' by the trace mechanism.
                    # In order to always generate sufficient events, we add the "assert
                    # TRUE" event to one slot, which fires every cycle, and thus fills our
                    # buffer quickly.

                    # Some events:
                    # TRUE                       (0x01)
                    # STREAM_STALL               (0x18)
                    # LOCK_STALL                 (0x1A)
                    # EVENTS_CORE_INSTR_EVENT_1  (0x22)
                    # EVENTS_CORE_INSTR_EVENT_0  (0x21)
                    # INSTR_VECTOR               (0x25)  Core executes a vecotr MAC, ADD or compare instruction
                    # INSTR_LOCK_ACQUIRE_REQ     (0x2C)  Core executes a lock acquire instruction
                    # INSTR_LOCK_RELEASE_REQ     (0x2D)  Core executes a lock release instruction
                    # EVENTS_CORE_PORT_RUNNING_1 (0x4F)
                    # EVENTS_CORE_PORT_RUNNING_0 (0x4B)

                    # Trace_Event0  (4 slots)
                    IpuWrite32(0, 2, 0x340E0, 0x4B222125)
                    # Trace_Event1  (4 slots)
                    IpuWrite32(0, 2, 0x340E4, 0x2D2C1A4F)

                    # Event slots as configured above:
                    # 0: Kernel executes vector instruction
                    # 1: Event 0 -- Kernel starts
                    # 2: Event 1 -- Kernel done
                    # 3: Port_Running_0
                    # 4: Port_Running_1
                    # 5: Lock Stall
                    # 6: Lock Acquire Instr
                    # 7: Lock Release Instr

                    # Stream_Switch_Event_Port_Selection_0
                    # This is necessary to capture the Port_Running_0 and Port_Running_1 events
                    IpuWrite32(0, 2, 0x3FF00, 0x121)

                    # Trace_Control0: Define trace start and stop triggers. Set start event TRUE.
                    IpuWrite32(0, 2, 0x340D0, 0x10000)

                    # Start trace copy out.
                    IpuWriteBdShimTile(
                        bd_id=3,
                        buffer_length=traceSizeInBytes,
                        buffer_offset=tensorSize,
                        enable_packet=0,
                        out_of_order_id=0,
                        packet_id=0,
                        packet_type=0,
                        column=0,
                        column_num=1,
                        d0_stride=0,
                        d0_wrap=0,
                        d1_stride=0,
                        d1_wrap=0,
                        d2_stride=0,
                        ddr_id=2,
                        iteration_current=0,
                        iteration_stride=0,
                        iteration_wrap=0,
                        lock_acq_enable=0,
                        lock_acq_id=0,
                        lock_acq_val=0,
                        lock_rel_id=0,
                        lock_rel_val=0,
                        next_bd=0,
                        use_next_bd=0,
                        valid_bd=1,
                    )
                    IpuWrite32(0, 0, 0x1D20C, 0x3)

                ipu_dma_memcpy_nd(
                    metadata="in",
                    bd_id=0,
                    mem=inTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_dma_memcpy_nd(
                    metadata="out",
                    bd_id=1,
                    mem=outTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


passthroughKernel()
