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

                # Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
                if enable_tracing:
                    # 0x340D0: Trace Control 0
                    #          0xAABB---C
                    #            AA        <- Event to stop trace capture
                    #              BB      <- Event to start trace capture
                    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
                    # Configure so that "Event 1" (always true) causes tracing to start
                    ipu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340D0,
                        value=0x00010000,
                    )
                    # 0x340D4: Trace Control 1
                    ipu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340D4,
                        value=0x00000000,
                    )
                    # 0x340E0: Trace Event Group 1  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340E0,
                        value=0x4B222125,
                    )
                    # 0x340E4: Trace Event Group 2  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340E4,
                        value=0x2D2C1A4F,
                    )

                    ipu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x3FF00,
                        value=0x00000121,
                    )

                    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
                    # out to host DDR memory
                    trace_bd_id = 13  # use BD 13 for writing trace output from compute tile to DDR host memory
                    output_size = N_in_bytes
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
                    # Set start BD to our shim bd_Id (3)
                    ipu_write32(column=0, row=0, address=0x1D20C, value=trace_bd_id)

                ipu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_vector_scalar()
