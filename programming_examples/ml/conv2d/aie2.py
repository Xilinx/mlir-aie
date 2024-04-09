#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

width = 32
height = 32
in_channels = 64
out_channels = 64

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


actIn = width * in_channels  # 32*64 = 2048
bufIn = actIn * 2
actInInt32s = actIn // 4

weights = in_channels * out_channels
weightsInInt32s = weights // 4

enableTrace = True
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():

            actIn_ty = T.memref(actIn, T.i8())
            bufIn_ty = T.memref(bufIn, T.i8())
            weights_ty = T.memref(weights, T.i8())
            bufOut_ty = T.memref(bufIn, T.ui8())
            out_ty = T.memref(actIn, T.ui8())

            # memRef_3x3_ty = T.memref(3, 3, T.i16())

            ofifo_actIn_ty = TypeAttr.get(ObjectFifoType.get(actIn_ty))
            ofifo_weights_ty = TypeAttr.get(ObjectFifoType.get(weights_ty))
            ofifo_bufIn_ty = TypeAttr.get(ObjectFifoType.get(bufIn_ty))
            ofifo_out_ty = TypeAttr.get(ObjectFifoType.get(out_ty))
            ofifo_bufOut_ty = TypeAttr.get(ObjectFifoType.get(bufOut_ty))

            # AIE Core Function declarations
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    T.i32(),
                    T.i32(),
                    T.i32(),
                    T.i32(),
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            # ComputeTile3 = tile(0, 3)
            # ComputeTile4 = tile(0, 4)
            # ComputeTile5 = tile(0, 5)

            if enableTrace:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)
                # flow(ComputeTile2, "Trace", 0, ShimTile, "DMA", 1)

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2=object_fifo(
                "inOF_act_L3L2",
                ShimTile,
                MemTile,
                2,
                bufIn_ty
            )
            of_act_L2_02=object_fifo(
                "act_L2_02",
                MemTile,
                ComputeTile2,
                2,
                actIn_ty
            )
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2=object_fifo(
                "inOF_wts_0_L3L2",
                ShimTile,
                [ComputeTile2],
                1,
                weights_ty
            )

            # Output
            of_out_02_L2=object_fifo(
                "out_02_L2",
                ComputeTile2,
                [MemTile],
                2,
                out_ty
            )
            of_outOFL2L3=object_fifo(
                "outOFL2L3",
                MemTile,
                [ShimTile],
                2,
                bufOut_ty
            )
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up compute tiles

            rtp2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                y_dim = 32
                x_dim = 32
                ci = 64
                co = 64

                for _ in for_(4294967295):
                    elemWts = of_inOF_wts_0_L3L2.acquire(
                        ObjectFifoPort.Consume, 1)

                    scale = memref.load(rtp2, [0])
                    # scale = memref.load(rtpComputeTile2, [0])

                    for _ in for_(y_dim):
                        elemIn = of_act_L2_02.acquire(
                            ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(
                            ObjectFifoPort.Produce, 1)

                        call(
                            conv2dk1_i8,
                            [
                                elemIn,
                                elemWts,
                                elemOut0,
                                arith.constant(x_dim),
                                arith.constant(ci),
                                arith.constant(co),
                                scale,
                            ],
                        )

                        objectfifo_release(ObjectFifoPort.Consume, "act_L2_02", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out_02_L2", 1)
                        yield_([])
                    objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_0_L3L2", 1)
                    yield_([])

            # To/from AIE-array data movement

            tensorSize = width * height * in_channels
            tensorSizeInInt32s = tensorSize // 4
            tensor_ty = T.memref(tensorSizeInInt32s, T.i32())
            memRef_wts_ty = T.memref(weightsInInt32s, T.i32())
            # memRef_16x16_ty = T.memref(16, 16, T.i32())

            @FuncOp.from_py_func(tensor_ty, memRef_wts_ty, tensor_ty)
            def sequence(I, W, O):
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
                    ipu_write32(0, 2, 0x340E0, 0x4B222125)
                    # Trace_Event1  (4 slots)
                    ipu_write32(0, 2, 0x340E4, 0x2D2C1A4F)

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
                    ipu_write32(0, 2, 0x3FF00, 0x121)

                    # Trace_Control0: Define trace start and stop triggers. Set start event TRUE.
                    ipu_write32(0, 2, 0x340D0, 0x10000)

                    # Start trace copy out.
                    ipu_writebd_shimtile(
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
                        # d0_wrap=0,
                        d0_size=0,
                        d1_stride=0,
                        # d1_wrap=0,
                        d1_size=0,
                        d2_stride=0,
                        ddr_id=2,
                        iteration_current=0,
                        iteration_stride=0,
                        # iteration_wrap=0,
                        iteration_size=0,
                        lock_acq_enable=0,
                        lock_acq_id=0,
                        lock_acq_val=0,
                        lock_rel_id=0,
                        lock_rel_val=0,
                        next_bd=0,
                        use_next_bd=0,
                        valid_bd=1,
                    )
                    ipu_write32(0, 0, 0x1D20C, 0x3)

                IpuWriteRTPOp("rtp2", col=0, row=2, index=0, value=9)

                ipu_dma_memcpy_nd(
                    metadata="inOF_act_L3L2",
                    bd_id=0,
                    mem=I,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=2,
                    mem=O,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                ipu_dma_memcpy_nd(
                    metadata="inOF_wts_0_L3L2",
                    bd_id=2,
                    mem=W,
                    sizes=[1, 1, 1, weightsInInt32s],
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


conv2dk1()
