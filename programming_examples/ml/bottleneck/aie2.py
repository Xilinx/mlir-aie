#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_
from aie.extras.util import np_ndarray_type_get_shape

# tracing definitions
trace_sz_in_bytes = 8192
trace_sz_in_i32s = trace_sz_in_bytes // 4
enableTrace = False

# Define bottleneck layer sizes

tensorInW = 32
tensorInH = 32
tensorInC = 256

tensorL1InC = tensorInC
tensorL1OutC = tensorL1InC // 4

tensorL2InC = tensorL1OutC
tensorL2OutC = tensorL2InC

tensorL3InC = tensorL2OutC
tensorL3OutC = tensorL3InC * 4

activationsIn = tensorInW * tensorInH * tensorInC
acitivationsOut = activationsIn
totalWeights = (
    tensorL1InC * tensorL1OutC
    + 3 * 3 * tensorL2InC * tensorL2OutC
    + tensorL3InC * tensorL3OutC
)


def bottleneck4AIEs():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def deviceBody():

            # define types
            activationsInL3_ty = np.ndarray[(activationsIn,), np.dtype[np.int8]]
            weightsInL3_ty = np.ndarray[(totalWeights,), np.dtype[np.uint8]]
            weightsAll_ty = np.ndarray[(totalWeights,), np.dtype[np.int8]]

            tensorLayer1In_ty = np.ndarray[
                (tensorInW, 1, tensorL1InC), np.dtype[np.int8]
            ]
            weightsLayer1_ty = np.ndarray[
                (tensorL1InC * tensorL1OutC,), np.dtype[np.int8]
            ]
            tensorLayer1Out_ty = np.ndarray[
                (tensorInW, 1, tensorL1OutC), np.dtype[np.uint8]
            ]

            tensorLayer2In_ty = np.ndarray[
                (tensorInW, 1, tensorL2InC), np.dtype[np.uint8]
            ]
            weightsLayer2_ty = np.ndarray[
                (3 * 3 * tensorL2InC * tensorL2OutC,), np.dtype[np.int8]
            ]
            tensorLayer2Out_ty = np.ndarray[
                (tensorInW, 1, tensorL2OutC // 2), np.dtype[np.uint8]
            ]

            tensorLayer3In_ty = np.ndarray[
                (tensorInW, 1, tensorL3InC // 2), np.dtype[np.uint8]
            ]
            weightsLayer3_ty = np.ndarray[
                (tensorL3InC * tensorL3OutC,), np.dtype[np.int8]
            ]
            tensorLayer3Out_ty = np.ndarray[
                (tensorInW, 1, tensorL3OutC), np.dtype[np.uint8]
            ]

            # kernel definitions
            conv2dk1 = external_func(
                "conv2dk1_i8",
                inputs=[
                    tensorLayer1In_ty,
                    weightsLayer1_ty,
                    tensorLayer1Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk3 = external_func(
                "conv2dk3_ui8",
                inputs=[
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    weightsLayer2_ty,
                    tensorLayer2Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk1_skip = external_func(
                "conv2dk1_skip_i8",
                inputs=[
                    tensorLayer3In_ty,
                    tensorLayer3In_ty,
                    weightsLayer3_ty,
                    tensorLayer3Out_ty,
                    tensorLayer1In_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            if enableTrace:
                flow(ComputeTile4, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # runtime parameters

            rtpComputeTile2 = buffer(
                ComputeTile2, np.ndarray[(16), np.int32], "rtpComputeTile2"
            )
            rtpComputeTile3 = buffer(
                ComputeTile3, np.ndarray[(16), np.int32], "rtpComputeTile3"
            )
            rtpComputeTile4 = buffer(
                ComputeTile4, np.ndarray[(16), np.int32], "rtpComputeTile4"
            )
            rtpComputeTile5 = buffer(
                ComputeTile5, np.ndarray[(16), np.int32], "rtpComputeTile5"
            )

            # set up data movement with OFs
            # input tensor (with broadcast for skip connection)
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2",
                ShimTile,
                [ComputeTile2, MemTile],
                [2, 2, 4],
                tensorLayer1In_ty,
            )
            of_skip_buf = object_fifo(
                "skip_buf", MemTile, ComputeTile4, 2, tensorLayer1In_ty
            )
            object_fifo_link(of_inOF_act_L3L2, of_skip_buf)

            # weights
            inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, MemTile, 1, weightsAll_ty
            )
            of_wts_buf_00 = object_fifo(
                "wts_buf_00", MemTile, ComputeTile2, 1, weightsLayer1_ty
            )
            wts_buf_01 = object_fifo(
                "wts_buf_01",
                MemTile,
                [ComputeTile3, ComputeTile5],
                1,
                weightsLayer2_ty,
            )
            wts_buf_02 = object_fifo(
                "wts_buf_02", MemTile, ComputeTile4, 1, weightsLayer3_ty
            )
            of_offsets = [
                0,
                np.prod(np_ndarray_type_get_shape(weightsLayer1_ty)),
                np.prod(np_ndarray_type_get_shape(weightsLayer1_ty))
                + np.prod(np_ndarray_type_get_shape(weightsLayer2_ty)),
            ]
            object_fifo_link(
                inOF_wts_0_L3L2, [of_wts_buf_00, wts_buf_01, wts_buf_02], [], of_offsets
            )

            # activation tensor
            of_act_2_3_5 = object_fifo(
                "act_2_3_5",
                ComputeTile2,
                [ComputeTile3, ComputeTile5],
                [2, 4, 4],
                tensorLayer1Out_ty,
            )  # 1x1 -> 3x3
            act_3_4 = object_fifo(
                "act_3_4", ComputeTile3, ComputeTile4, 2, tensorLayer2Out_ty
            )  # 3x3 -> 1x1
            act_5_4 = object_fifo(
                "act_5_4", ComputeTile5, ComputeTile4, 2, tensorLayer2Out_ty
            )  # 3x3 -> 1x1

            # output tensor
            outOFL2L3 = object_fifo(
                "outOFL2L3", ComputeTile4, ShimTile, 2, tensorLayer3Out_ty
            )

            # 1x1 conv2d
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    # acquire weights once
                    element0Weights = of_wts_buf_00.acquire(ObjectFifoPort.Consume, 1)
                    scale = rtpComputeTile2[0]
                    for _ in range_(tensorInH):
                        element0ActivactionsIn = of_inOF_act_L3L2.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = of_act_2_3_5.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk1(
                            element0ActivactionsIn,
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL1InC,
                            tensorL1OutC,
                            scale,
                        )
                        of_inOF_act_L3L2.release(ObjectFifoPort.Consume, 1)
                        of_act_2_3_5.release(ObjectFifoPort.Produce, 1)
                    of_wts_buf_00.release(ObjectFifoPort.Consume, 1)

            # 3x3 conv2d OFM 0-31
            @core(ComputeTile3, "conv2dk3.o")
            def core_body():
                scale = 11
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile3, 0)

                    # pre-amble: top row
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    )
                    act_3_4.release(ObjectFifoPort.Produce, 1)

                    # middle
                    for _ in range_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_3_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL2InC,
                            tensorL2OutC,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        )
                        of_act_2_3_5.release(ObjectFifoPort.Consume, 1)
                        act_3_4.release(ObjectFifoPort.Produce, 1)

                    # last part
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    )

                    of_act_2_3_5.release(ObjectFifoPort.Consume, 2)
                    act_3_4.release(ObjectFifoPort.Produce, 1)
                    wts_buf_01.release(ObjectFifoPort.Consume, 1)

            # 3x3 conv2d OFM 32-63
            @core(ComputeTile5, "conv2dk3.o")
            def core_body():
                scale = 11
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile5, 0)

                    # pre-amble: top row
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_5_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        0,
                        scale,
                        tensorL2OutC // 2,
                    )
                    act_5_4.release(ObjectFifoPort.Produce, 1)

                    # middle
                    for _ in range_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_5_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL2InC,
                            tensorL2OutC,
                            3,
                            3,
                            1,
                            scale,
                            tensorL2OutC // 2,
                        )
                        of_act_2_3_5.release(ObjectFifoPort.Consume, 1)
                        act_5_4.release(ObjectFifoPort.Produce, 1)

                    # last part
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_5_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        2,
                        scale,
                        tensorL2OutC // 2,
                    )
                    of_act_2_3_5.release(ObjectFifoPort.Consume, 2)
                    act_5_4.release(ObjectFifoPort.Produce, 1)
                    wts_buf_01.release(ObjectFifoPort.Consume, 1)

            # # 1x1 conv2d and add skip
            @core(ComputeTile4, "conv2dk1_skip.o")
            def core_body():
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_02.acquire(ObjectFifoPort.Consume, 1)
                    scale = rtpComputeTile4[0]
                    skipScale = rtpComputeTile4[1]

                    for _ in range_(tensorInH):
                        element0ActivactionsIn = act_3_4.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element1ActivactionsIn = act_5_4.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        elementSkipsIn = of_skip_buf.acquire(ObjectFifoPort.Consume, 1)
                        elementActivactionsOut = outOFL2L3.acquire(
                            ObjectFifoPort.Produce, 1
                        )

                        conv2dk1_skip(
                            element0ActivactionsIn,
                            element1ActivactionsIn,
                            element0Weights,
                            elementActivactionsOut,
                            elementSkipsIn,
                            tensorInW,
                            tensorL3InC,
                            tensorL3OutC,
                            scale,
                            skipScale,
                        )
                        outOFL2L3.release(ObjectFifoPort.Produce, 1)
                        act_3_4.release(ObjectFifoPort.Consume, 1)
                        act_5_4.release(ObjectFifoPort.Consume, 1)
                        of_skip_buf.release(ObjectFifoPort.Consume, 1)
                    wts_buf_02.release(ObjectFifoPort.Consume, 1)

            # instruction stream generation
            @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsInL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):

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
                    # INSTR_LOCK_ACQUIRE_REQ     (0x2C)  Core executes a lock .acquire instruction
                    # INSTR_LOCK_.release_REQ     (0x2D)  Core executes a lock .release instruction
                    # EVENTS_CORE_PORT_RUNNING_1 (0x4F)
                    # EVENTS_CORE_PORT_RUNNING_0 (0x4B)

                    # Trace_Event0  (4 slots)
                    npu_write32(0, 4, 0x340E0, 0x4B222125)
                    # Trace_Event1  (4 slots)
                    npu_write32(0, 4, 0x340E4, 0x2D2C1A4F)

                    # Event slots as configured above:
                    # 0: Kernel executes vector instruction
                    # 1: Event 0 -- Kernel starts
                    # 2: Event 1 -- Kernel done
                    # 3: Port_Running_0
                    # 4: Port_Running_1
                    # 5: Lock Stall
                    # 6: Lock .acquire Instr
                    # 7: Lock .release Instr

                    # Stream_Switch_Event_Port_Selection_0
                    # This is necessary to capture the Port_Running_0 and Port_Running_1 events
                    npu_write32(0, 4, 0x3FF00, 0x121)

                    # Trace_Control0: Define trace start and stop triggers. Set start event TRUE.
                    npu_write32(0, 4, 0x340D0, 0x10000)

                    # Start trace copy out.
                    npu_writebd(
                        bd_id=3,
                        buffer_length=trace_sz_in_i32s,
                        buffer_offset=acitivationsOut,
                        enable_packet=0,
                        out_of_order_id=0,
                        packet_id=0,
                        packet_type=0,
                        column=0,
                        row=0,
                        d0_stepsize=0,
                        d0_wrap=0,
                        d1_stepsize=0,
                        d1_wrap=0,
                        d2_stepsize=0,
                        iteration_current=0,
                        iteration_stepsize=0,
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
                    npu_write32(0, 2, 0x1D20C, 0x3)

                # write RTP parameters
                NpuWriteRTPOp("rtpComputeTile2", index=0, value=1)  # scale
                NpuWriteRTPOp("rtpComputeTile3", index=0, value=1)  # scale
                NpuWriteRTPOp("rtpComputeTile5", index=0, value=1)  # scale
                NpuWriteRTPOp(
                    "rtpComputeTile4", index=0, value=1
                )  # scale: conv1x1 with the same scale as the input so we match the scaling factor of output after conv1x1 and the initial input
                NpuWriteRTPOp("rtpComputeTile4", index=1, value=0)  # skip_scale

                npu_dma_memcpy_nd(
                    metadata=of_inOF_act_L3L2,
                    bd_id=0,
                    mem=inputFromL3,
                    sizes=[1, 1, 1, activationsIn],
                )
                npu_dma_memcpy_nd(
                    metadata=outOFL2L3,
                    bd_id=2,
                    mem=outputToL3,
                    sizes=[1, 1, 1, acitivationsOut],
                )
                npu_dma_memcpy_nd(
                    metadata=inOF_wts_0_L3L2,
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeights],
                )
                # outOFL2L3 will only complete after inOF_wts_0_L3L2 and of_inOF_act_L3L2 complete, so we just wait on outOFL2L3 instead of all
                dma_wait(outOFL2L3)

    print(ctx.module)


bottleneck4AIEs()
