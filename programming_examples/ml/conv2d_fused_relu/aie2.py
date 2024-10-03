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

width = 32
height = 32
in_channels = 64
out_channels = 64

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


actIn = width * in_channels  # 32*64 = 2048
bufIn = actIn * 2  # double buffer

weights = in_channels * out_channels

actOut = width * out_channels  # 32*64 = 2048
bufOut = actOut * 2  # double buffer

tensorSize = width * height * in_channels

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
            bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]

            weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

            out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
            bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]

            tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]

            # AIE Core Function declarations
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            compute_tile2_col, compute_tile2_row = 0, 2

            if enableTrace:
                flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, bufIn_ty
            )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 2, actIn_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 1, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2, out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, bufOut_ty)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up compute tiles

            rtp2 = buffer(ComputeTile2, T.memref(16, T.i32()), "rtp2")

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                y_dim = 32
                x_dim = 32
                ci = 64
                co = 64

                for _ in range_(0xFFFFFFFF):
                    elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                    scale = rtp2[0]
                    # scale = memref.load(rtpComputeTile2, [0])

                    for _ in range_(y_dim):
                        elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                        conv2dk1_i8(elemIn, elemWts, elemOut0, x_dim, ci, co, scale)

                        of_act_L2_02.release(ObjectFifoPort.Consume, 1)
                        of_out_02_L2.release(ObjectFifoPort.Produce, 1)
                    of_inOF_wts_0_L3L2.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement

            @runtime_sequence(tensor_ty, weights_ty, tensor_ty)
            def sequence(I, W, O):
                if enableTrace:
                    # 0x340D0: Trace Control 0
                    #          0xAABB---C
                    #            AA        <- Event to stop trace capture
                    #              BB      <- Event to start trace capture
                    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
                    # Configure so that "Event 1" (always true) causes tracing to start
                    npu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340D0,
                        value=0x00010000,
                    )
                    # 0x340D4: Trace Control 1
                    npu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340D4,
                        value=0x00000000,
                    )
                    # 0x340E0: Trace Event Group 1  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    npu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340E0,
                        value=0x4B222125,
                    )
                    # 0x340E4: Trace Event Group 2  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    npu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x340E4,
                        value=0x2D2C1A4F,
                    )

                    npu_write32(
                        column=compute_tile2_col,
                        row=compute_tile2_row,
                        address=0x3FF00,
                        value=0x00000121,
                    )

                    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
                    # out to host DDR memory
                    trace_bd_id = 13  # use BD 13 for writing trace output from compute tile to DDR host memory
                    output_size = bufOut
                    npu_writebd(
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
                    npu_write32(column=0, row=0, address=0x1D20C, value=trace_bd_id)

                NpuWriteRTPOp("rtp2", index=0, value=1)

                npu_dma_memcpy_nd(
                    metadata=of_inOF_act_L3L2,
                    bd_id=0,
                    mem=I,
                    sizes=[1, 1, 1, tensorSize],
                )
                npu_dma_memcpy_nd(
                    metadata=of_outOFL2L3,
                    bd_id=2,
                    mem=O,
                    sizes=[1, 1, 1, tensorSize],
                )
                npu_dma_memcpy_nd(
                    metadata=of_inOF_wts_0_L3L2,
                    bd_id=2,
                    mem=W,
                    sizes=[1, 1, 1, weights],
                )
                # of_outOFL2L3 will only complete after of_inOF_wts_0_L3L2 and of_inOF_act_L3L2 complete, so we just wait on of_outOFL2L3 instead of all
                dma_wait(of_outOFL2L3)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


conv2dk1()
