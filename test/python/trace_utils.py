#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.

# RUN: %python %s | FileCheck %s

# Test that configure_trace() and start_trace() generate correct aie.trace ops

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T

import aie.utils.trace as trace_utils
from aie.utils.trace.events import (
    PortEvent,
    CoreEvent,
    MemEvent,
)

N = 1024

if len(sys.argv) == 2:
    N = int(sys.argv[1])

lineWidthInBytes = N // 4  # chop input in 4 sub-tensors
lineWidthInInt32s = lineWidthInBytes // 4


def passthroughKernel():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            # define types
            memRef_ty = T.memref(lineWidthInBytes, T.ui8())

            # AIE Core Function declarations
            passThroughLine = external_func(
                "passThroughLine",
                inputs=[memRef_ty, memRef_ty, T.i32()],
                link_with="passThrough.cc.o",
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in.acquire(ObjectFifoPort.Consume, 1)
                    passThroughLine(elemIn, elemOut, lineWidthInBytes)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

            # Configure tracing with custom events on multiple tiles
            # CHECK: aie.trace @trace_core_1(%{{.*}}) {
            # CHECK:   aie.trace.mode "Event-Time"
            # CHECK:   aie.trace.packet id = 1 type = core
            # CHECK:   aie.trace.event <"INSTR_EVENT_1">
            # CHECK:   aie.trace.event <"INSTR_EVENT_0">
            # CHECK:   aie.trace.event <"INSTR_VECTOR">
            # CHECK:   aie.trace.event <"PORT_RUNNING_0">
            # CHECK:   aie.trace.event <"PORT_RUNNING_1">
            # CHECK:   aie.trace.event <"INSTR_LOCK_ACQUIRE_REQ">
            # CHECK:   aie.trace.event <"LOCK_STALL">
            # CHECK:   aie.trace.event <"MEMORY_STALL">
            # CHECK:   aie.trace.port<0> port = DMA channel = 0 direction = S2MM
            # CHECK:   aie.trace.port<1> port = DMA channel = 0 direction = MM2S
            # CHECK:   aie.trace.start broadcast = 15
            # CHECK:   aie.trace.stop broadcast = 14
            # CHECK: }
            # CHECK: aie.trace @trace_shim_2(%{{.*}}) {
            # CHECK:   aie.trace.packet id = 2 type = shimtile
            # CHECK:   aie.trace.event <"DMA_S2MM_0_START_TASK">
            # CHECK:   aie.trace.event <"DMA_S2MM_1_START_TASK">
            # CHECK:   aie.trace.event <"DMA_MM2S_0_START_TASK">
            # CHECK:   aie.trace.event <"DMA_S2MM_0_FINISHED_TASK">
            # CHECK:   aie.trace.event <"DMA_S2MM_1_FINISHED_TASK">
            # CHECK:   aie.trace.event <"DMA_MM2S_0_FINISHED_TASK">
            # CHECK:   aie.trace.event <"DMA_S2MM_0_STREAM_STARVATION">
            # CHECK:   aie.trace.event <"DMA_S2MM_1_STREAM_STARVATION">
            # CHECK:   aie.trace.start broadcast = 15
            # CHECK:   aie.trace.stop broadcast = 14
            # CHECK: }
            trace_utils.configure_trace(
                [ComputeTile2, ShimTile],
                coretile_events=[
                    CoreEvent.INSTR_EVENT_1,
                    CoreEvent.INSTR_EVENT_0,
                    CoreEvent.INSTR_VECTOR,
                    PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
                    PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False),
                    CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
                    CoreEvent.LOCK_STALL,
                    CoreEvent.MEMORY_STALL,
                ],
            )

            tensorSize = N
            tensorSizeInInt32s = tensorSize // 4
            tensor_ty = T.memref(lineWidthInInt32s, T.i32())

            # Verify start_trace() generates host_config and start_config ops
            # CHECK: aie.runtime_sequence
            # CHECK:   aie.trace.host_config buffer_size = 8192
            # CHECK:   aie.trace.start_config @trace_core_1
            # CHECK:   aie.trace.start_config @trace_shim_2
            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(inTensor, outTensor, notUsed):
                trace_utils.start_trace(trace_size=8192, ddr_id=4)

                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=0,
                    mem=inTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=1,
                    mem=outTensor,
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                dma_wait(of_out)

    print(ctx.module)


passthroughKernel()
