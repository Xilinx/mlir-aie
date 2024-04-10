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

def pack4bytes(b3, b2, b1, b0):
    w = (b3 & 0xFF) << 24
    w |= (b2 & 0xFF) << 16
    w |= (b1 & 0xFF) << 8
    w |= (b0 & 0xFF) << 0
    return w

# Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
# This is a very simple model of tracing, which has some big assumptions:
# 1) Trace data is collected over circuit switched connections, not packet-switched
# 2) A ShimDMA S2MM channel is dedicated to the trace data
# 3) Trace data is small enough to fit in a fixed-size buffer, which is collected with the
# outputs of the design
# 4) The usual model of '2 inputs, 1 output' is followed, and the
# trace data is appended to the other outputs

# tile: The tile we're tracing
# shim: The shim tile to output data with.
# bd_id: The BD in the shim tile to use.
# channel: The S2MM channel to use (0 or 1).
# size: The size of the trace data
# offset: The offset of the trace data in the (single) output buffer.
# start: The event number to start tracing on
# stop: The event number to stop tracing on
# events: A list of events to trace.  Up to 8 events are allowed in aie2, more are ignored

# Event numbers should be less than 128.
# Big assumption: The bd_id and channel are unused.  If they are used by something else, then
# everything will probably break.
def configure_simple_tracing_aie2(tile, shim, bd_id, channel, size, offset, start, stop, events):
    assert(shim.isShimTile())

    # Pad the input so we have exactly 8 events.
    events = (events + [0] * 8)[:8]

    # 0x340D0: Trace Control 0
    #          0xAABB---C
    #            AA        <- Event to stop trace capture
    #              BB      <- Event to start trace capture
    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
    # Configure so that "Event 1" (always true) causes tracing to start
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x340D0,
        value=pack4bytes(stop, start, 0, 0),
    )
    # 0x340D4: Trace Control 1
    # This is used to control packet routing.  For the moment
    # only deal with the simple case of circuit routing.
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x340D4,
        value=0,
    )
    # 0x340E0: Trace Event Group 1  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x340E0,
        value=pack4bytes(*events[0:3]),
    )
    # 0x340E4: Trace Event Group 2  (Which events to trace)
    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x340E4,
        value=pack4bytes(*events[4:7]),
    )

    # 0x3FF00: Stream switch event port selection 0
    def master(port):
        return port | (1 << 5)
    def slave(port):
        return port
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x3FF00,
        value=pack4bytes(0, 0, slave(1), master(1)), # port 1 is FIFO0?
    )
    ipu_write32(
        column=tile.col(),
        row=tile.row(),
        address=0x3FF04,
        value=pack4bytes(0, 0, 0, 0),
    )

    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
    # out to host DDR memory
    ipu_writebd_shimtile(
        bd_id=bd_id,
        buffer_length=size,
        buffer_offset=offset,
        enable_packet=0,
        out_of_order_id=0,
        packet_id=0,
        packet_type=0,
        column=shim.col(),
        column_num=1,
        d0_size=0,
        d0_stride=0,
        d1_size=0,
        d1_stride=0,
        d2_stride=0,
        # Assume using output buffer.  This probably needs to be configurable.
        ddr_id=2, 
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
    # configure S2MM channel
    ipu_write32(column=shim.col(), row=shim.row(), address=0x1D204 if channel == 0 else 0x1D20C, value=bd_id)

def my_vector_scalar():
    N = 4096
    N_in_bytes = N * 4
    n = 1024
    N_div_n = N // n

    buffer_depth = 2

    vectorized = True
    enable_tracing = False
    trace_size = 8192

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(n, T.i32())

            # AIE Core Function declarations

            scale_scalar_int32 = external_func(
                "scale_scalar_int32", inputs=[memRef_ty, memRef_ty]
            )
            scale_int32 = external_func("scale_int32", inputs=[memRef_ty, memRef_ty])

            # Tile declarations
            ShimTile = tile(0, 0)
            compute_tile2_col, compute_tile2_row = 0, 2
            ComputeTile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2, "scale.o")
            def core_body():
                # Effective while(1)
                for _ in for_(sys.maxsize):
                    # Number of sub-vector "tile" iterations
                    for _ in for_(N_div_n):
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                        if vectorized:
                            call(scale_int32, [elem_in, elem_out])
                        else:
                            call(scale_scalar_int32, [elem_in, elem_out])
                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):

                if enable_tracing:
                    configure_simple_tracing(ComputeTile2,
                                             ShimTile,
                                             bd_id=13,
                                             channel=1,
                                             size=trace_size,
                                             offset=N_in_bytes,
                                             start=0x1,
                                             stop=0x0,
                                             events={0x4B, 0x22, 0x21, 0x25, 0x2D, 0x2C, 0x1A, 0x4F})

                ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
