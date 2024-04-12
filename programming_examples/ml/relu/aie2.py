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


def my_relu():

    word_size_in = 2
    N = 65536
    N_in_bytes = N * word_size_in

    A_sz_in_i32s = N_in_bytes // 4
    C_sz_in_i32s = N_in_bytes // 4

    enable_tracing = True
    trace_size = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores
    buffer_depth = 2

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(n, T.bf16())

            # Type used in the tile memory
            memRef_A_ty = T.memref(n, T.bf16())
            memRef_C_ty = T.memref(n, T.bf16())

            # Type used in the memory tile which aggregates across the 4 cores
            memRef_A_MT_ty = T.memref(n * n_cores, T.bf16())
            memRef_C_MT_ty = T.memref(n * n_cores, T.bf16())

            # AIE Core Function declarations

            bf16_relu = external_func("bf16_relu", inputs=[memRef_ty, memRef_ty])

            # Tile declarations
            ShimTile = tile(0, 0)

            MemTile = tile(0, 1)
            cores = [tile(0, 2 + i) for i in range(n_cores)]

            inA_fifo_names = [f"memA{i}" for i in range(n_cores)]
            outC_fifo_names = [f"memC{i}" for i in range(n_cores)]

            inA_fifos = {}
            outC_fifos = {}

            # AIE-array data movement with object fifos
            # Input A
            inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, memRef_A_MT_ty)
            for i in range(n_cores):
                inA_fifos[inA_fifo_names[i]] = object_fifo(
                    inA_fifo_names[i], MemTile, cores[i], buffer_depth, memRef_A_ty
                )
            object_fifo_link(inA, inA_fifo_names)

            # Output C
            for i in range(n_cores):
                outC_fifos[outC_fifo_names[i]] = object_fifo(
                    outC_fifo_names[i], cores[i], MemTile, buffer_depth, memRef_C_ty
                )
            outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, memRef_C_MT_ty)
            object_fifo_link(outC_fifo_names[0:n_cores], outC)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(cores[0], WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], "bf16_relu.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        for _ in for_(tiles):
                            elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                                ObjectFifoPort.Consume, 1
                            )

                            call(bf16_relu, [elem_in_a, elem_out])

                            inA_fifos[inA_fifo_names[i]].release(
                                ObjectFifoPort.Consume, 1
                            )
                            outC_fifos[outC_fifo_names[i]].release(
                                ObjectFifoPort.Produce, 1
                            )
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
                        address=0x340D0,
                        value=0x00010000,
                    )
                    # 0x340D4: Trace Control 1
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340D4,
                        value=0x00000000,
                    )
                    # 0x340E0: Trace Event Group 1  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340E0,
                        value=0x00222100,
                    )
                    # 0x340E4: Trace Event Group 2  (Which events to trace)
                    #          0xAABBCCDD    AA, BB, CC, DD <- four event slots
                    ipu_write32(
                        column=0,
                        row=2,
                        address=0x340E4,
                        value=0x00000000,
                    )

                    ipu_write32(
                        column=0,
                        row=2,
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
                    # Set start BD to our shim bd_Id (13)
                    ipu_write32(column=0, row=0, address=0x1D20C, value=trace_bd_id)

                ipu_dma_memcpy_nd(
                    metadata="outC", bd_id=0, mem=C, sizes=[1, 1, 1, C_sz_in_i32s]
                )
                ipu_dma_memcpy_nd(
                    metadata="inA", bd_id=1, mem=A, sizes=[1, 1, 1, A_sz_in_i32s]
                )
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_relu()
