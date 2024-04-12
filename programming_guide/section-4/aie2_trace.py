#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.dialects.scf import *  # scf (strcutred control flow) dialect
from aie.extras.dialects.ext import memref, arith  # memref and arithmatic dialects


# AI Engine structural design function
def my_first_aie_program():

    enable_tracing = True
    trace_size = 8192
    C_sz_in_bytes = 64 * 4

    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # Dvice declaration - aie2 device IPU (aka Ryzen AI)
        @device(AIEDevice.ipu)
        def device_body():
            # Memref types
            memRef_8_ty = T.memref(8, T.i32())
            memRef_16_ty = T.memref(16, T.i32())
            memRef_32_ty = T.memref(32, T.i32())
            memRef_64_ty = T.memref(64, T.i32())

            # Tile declarations
            ComputeTile = tile(0, 2)
            ShimTile = tile(0, 0)

            compute_tile_col, compute_tile_row = 0, 2

            # Data movement with object FIFOs
            # Input (from shim tile to compute tile)
            of_in0 = object_fifo("in0", ShimTile, ComputeTile, 2, memRef_8_ty)

            # Output (from compute tile to shim tile)
            of_out0 = object_fifo("out0", ComputeTile, ShimTile, 2, memRef_8_ty)

            # Compute tile body
            @core(ComputeTile)
            def core_body():
                #                for _ in for_(0xFFFFFFFF):
                for _ in for_(8):
                    # Acquire input and output object FIFO objects
                    elem_in = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out0.acquire(ObjectFifoPort.Produce, 1)

                    # Core functionality - load, add 1, store
                    for i in for_(8):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])

                    # Release input and output object FIFO objects
                    of_in0.release(ObjectFifoPort.Consume, 1)
                    of_out0.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(ComputeTile, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # To/from AIE-array data movement
            @FuncOp.from_py_func(memRef_64_ty, memRef_64_ty, memRef_64_ty)
            def sequence(inTensor, notUsed, outTensor):

                # Configure tracing, see https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md
                if enable_tracing:
                    # 0x340D0: Trace Control 0
                    #          0xAABB---C
                    #            AA        <- Event to stop trace capture
                    #              BB      <- Event to start trace capture
                    #                   C  <- Trace mode, 00=event=time, 01=event-PC, 10=execution
                    # Configure so that "Event 1" (always true) causes tracing to start
                    ipu_write32(
                        column=compute_tile_col,
                        row=compute_tile_row,
                        address=0x340D0,
                        value=0x00010000,
                    )
                    # 0x340D4: Trace Control 1
                    ipu_write32(
                        column=compute_tile_col,
                        row=compute_tile_row,
                        address=0x340D4,
                        value=0x00000000,
                    )
                    # 0x340E0: Trace Event Group 1  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=compute_tile_col,
                        row=compute_tile_row,
                        address=0x340E0,
                        value=0x4B222125,
                    )
                    # 0x340E4: Trace Event Group 2  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=compute_tile_col,
                        row=compute_tile_row,
                        address=0x340E4,
                        value=0x2D2C1A4F,
                    )

                    ipu_write32(
                        column=compute_tile_col,
                        row=compute_tile_row,
                        address=0x3FF00,
                        value=0x00000121,
                    )

                    # Configure a buffer descriptor to write tracing information that has been routed into this shim tile
                    # out to host DDR memory
                    trace_bd_id = 13  # use BD 13 for writing trace output from compute tile to DDR host memory
                    output_size = C_sz_in_bytes
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

                ipu_dma_memcpy_nd(
                    metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, 64]
                )
                ipu_dma_memcpy_nd(
                    metadata="in0", bd_id=1, mem=inTensor, sizes=[1, 1, 1, 64]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    # Print the mlir conversion
    print(ctx.module)


# Call design function to generate mlir code to stdout
my_first_aie_program()
