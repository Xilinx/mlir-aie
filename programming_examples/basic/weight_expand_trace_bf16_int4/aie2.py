#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx


def my_expand():

    SF_BLOCK_SIZE = 32
    word_size_in = 2
    sf_word_size_in = 2
    N = 65536

    N_in_bytes = (N // word_size_in) + (N / SF_BLOCK_SIZE) * sf_word_size_in

    A_sz_in_i32s = (N // 8) + (
        N // SF_BLOCK_SIZE
    ) // 2  # They are 4 bits per element, we need to add on the scale factors later though
    B_sz_in_i32s = N // 2  # Returning 16 bits at the moment
    B_sz_in_bytes = N * 2  # Returning 16 bits at the moment

    # Tile sizes
    n = 1024
    block_size = 32
    sf_size = n // block_size

    input_buffer_size_bytes = (n // 2) + (
        sf_size * 2
    )  # They are bfloat16 sfs after the private values
    output_buffer_size_bytes = n * 2  # The unscaled values

    N_div_n = N // n

    n_cores = 1
    tiles = N_div_n // n_cores
    buffer_depth = 2

    enable_tracing = True
    trace_size = 16384



    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_i_ty = T.memref(
                input_buffer_size_bytes, T.i8()
            )  # Just think of the input as a raw byte buffer
            memRef_o_ty = T.memref(output_buffer_size_bytes, T.i8())  # For now

            # AIE Core Function declarations

            expand_int4_to_bfloat16 = external_func(
                "expand_int4_to_bfloat16", inputs=[memRef_i_ty, memRef_o_ty]
            )

            # Tile declarations
            ShimTile = tile(0, 0)

            MemTile = tile(0, 1)
            core0 = tile(0, 2)

            # AIE-array data movement with object fifos
            # Input
            inA = object_fifo("inA", ShimTile, core0, buffer_depth, memRef_i_ty)

            # Output B
            outB = object_fifo("outB", core0, ShimTile, buffer_depth, memRef_o_ty)


            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(core0, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles
            @core(core0, "expand.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles):
                        elem_out = outB.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = inA.acquire(ObjectFifoPort.Consume, 1)

                        call(expand_int4_to_bfloat16, [elem_in, elem_out])
                        inA.release(ObjectFifoPort.Consume, 1)
                        outB.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty)
            def sequence(A, C):


                # Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
                if enable_tracing:
                    # 0x340D0: Trace Control 0
                    #          0xAABB---C
                    #            AA        <- Event to stop trace capture
                    #              BB      <- Event to start trace capture
                    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
                    # Configure so that "Event 1" (always true) causes tracing to start
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340D0, # Trace Control 0, control of trace
                        value=0x00010000, # 30:24 event number to stop trace, 22:16 event number to start trace, 1:0 trace mode
                                            # 00 = event time, 01 = event PC, 10 = execution
                                            # binary: 0000 0000 0000 0001 0000 0000 0000 0000: start with 1, stop with 0, and use event time

                    )
                    # 0x340D4: Trace Control 1
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340D4, # Trace Control 1, control of trace: packet destination
                        value=0x00000000,
                    )
                    # 0x340E0: Trace Event Group 1  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    #          0x4B222125 is traditional

                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340E0, # trace event0: control of which internal event to trace
                        value=0x00222100, # 30:24, 22:16, 14:8, 6:0 are internal event numbers to add to trace
                        # binary: 0000 0000 0010 0010 0010 0001 0000 0000
                        # events: 0x22, 0x21, 0x00 are 
                    )
                    # 0x340E4: Trace Event Group 2  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    #          0x2D2C1A4F is traditional
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340E4, # trace event1: control of which internal event to trace
                        value=0x00000000, # 30:24, 22:16, 14:8, 6:0 are internal event number to add to trace
                    )

                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x3FF00, # Stream_Switch_Event_Port_Selection_0: Select Stream Switch Ports for event generation
                        value=0x00000121, # binary: 0000 0000 0000 0000 0000 0001 0010 0001:
                                            # Select port ID for port 0 event generation = 1, 
                                            # Select master or slave for port 0 event generation = 1 = master, 
                                            # Select port ID for port 1 event generation = 1,
                                            # Select master or slave for port 1 event generation = 0 = slave
                    )

                    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
                    # out to host DDR memory
                    trace_bd_id = 13  # use BD 13 for writing trace output from compute tile to DDR host memory
                    output_size = B_sz_in_bytes
                    ipu_writebd_shimtile(
                        bd_id=trace_bd_id,
                        buffer_length=trace_size,
                        buffer_offset=output_size,
                        enable_packet=0,
                        out_of_order_id=0,
                        packet_id=0,
                        packet_type=0,
                        column=0,
                        column_num=1,
                        d0_size=0,
                        d0_stride=0,
                        d1_size=0,
                        d1_stride=0,
                        d2_stride=0,
                        ddr_id=1,
                        iteration_current=0,
                        iteration_size=0,
                        iteration_stride=0,
                        lock_acq_enable=0,
                        lock_acq_id=0,
                        lock_acq_val=0,
                        lock_rel_id=0,
                        lock_rel_val=0,
                        next_bd=0,
                        use_next_bd=0,
                        valid_bd=1,
                    )
                    # Set start BD to our shim bd_Id (3)
                    ipu_write32(column=0, row=0, address=0x1D20C, value=trace_bd_id)

                ipu_dma_memcpy_nd(
                    metadata="outB", bd_id=0, mem=C, sizes=[1, 1, 1, B_sz_in_i32s]
                )
                ipu_dma_memcpy_nd(
                    metadata="inA", bd_id=1, mem=A, sizes=[1, 1, 1, A_sz_in_i32s]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_expand()