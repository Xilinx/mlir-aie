# aie2.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

width = 512 #1920 // 8
height = 9 #1080 // 8

lineWidthInBytes = width
lineWidthInInt32s = lineWidthInBytes // 4

enableTrace = False
traceSizeInBytes = 8192
traceSizeInInt32s = traceSizeInBytes // 4

@constructAndPrintInModule
def passThroughAIE2():
    @device("ipu")
    def deviceBody():
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int32_ty = IntegerType.get_signless(32)
        line_ty = MemRefType.get((lineWidthInBytes,), uint8_ty)

        passThroughLine = privateFunc("passThroughLine", inputs = [line_ty, line_ty, int32_ty], outputs = [])

        ShimTile = Tile(0, 0)
        ComputeTile2 = Tile(0, 2)

        if enableTrace:
            Flow(ComputeTile2, "Trace", 0, ShimTile, "DMA", 1)

        OrderedObjectBuffer("in", ShimTile, ComputeTile2, 2, line_ty)
        OrderedObjectBuffer("out", ComputeTile2, ShimTile, 2, line_ty)

        @core(ComputeTile2, "passThrough_aie2_8b.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 0XFFFFFFFF, step = 1)
            def loopReps():
                @forLoop(lowerBound = 0, upperBound =  height, step = 1)
                def loopTile():
                    elemOut = Acquire("out", "Produce", 1, line_ty).acquiredElem()  
                    elemIn = Acquire("in", "Consume", 1, line_ty).acquiredElem() 
                    call(passThroughLine, [elemIn, elemOut, width], [])
                    Release("in", "Consume", 1)
                    Release("out", "Produce", 1)


        tensorSize = width*height
        tensorSizeInInt32s = tensorSize // 4
        tensor_ty =  MemRefType.get((tensorSizeInInt32s,), int32_ty)
        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, notUsed, outTensor):
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
                IpuWriteBdShimTile(bd_id = 3,
                                   buffer_length = traceSizeInBytes,
                                   buffer_offset = tensorSize,
                                   enable_packet = 0,
                                   out_of_order_id = 0,
                                   packet_id = 0,
                                   packet_type = 0,
                                   column = 0,
                                   column_num = 1,
                                   d0_stepsize = 0,
                                   d0_wrap = 0,
                                   d1_stepsize = 0,
                                   d1_wrap = 0,
                                   d2_stepsize = 0,
                                   ddr_id = 2,
                                   iteration_current = 0,
                                   iteration_stepsize = 0,
                                   iteration_wrap = 0,
                                   lock_acq_enable = 0,
                                   lock_acq_id = 0,
                                   lock_acq_val = 0,
                                   lock_rel_id = 0,
                                   lock_rel_val = 0,
                                   next_bd = 0,
                                   use_next_bd = 0,
                                   valid_bd = 1)
                IpuWrite32(0, 0, 0x1D20C, 0x3)

            
            IpuDmaMemcpyNd(metadata = "in", bd_id = 1, mem = inTensor, lengths = [1, 1, 1, tensorSizeInInt32s]) 
            IpuDmaMemcpyNd(metadata = "out", bd_id = 0, mem = outTensor, lengths = [1, 1, 1, tensorSizeInInt32s]) 
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
