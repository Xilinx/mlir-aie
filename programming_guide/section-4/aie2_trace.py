#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir-aie context

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.dialects.scf import *  # scf (strcutred control flow) dialect
from aie.extras.dialects.ext import memref, arith  # memref and arithmatic dialects


# AI Engine structural design function
def my_first_aie_program():

    enableTrace = True
    trace_size = 8192
    C_sz_in_bytes = 64 * 4

    # Dvice declaration - aie2 device NPU
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

            if enableTrace:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile,
                    ShimTile,
                    channel=1,
                    bd_id=13,
                    ddr_id=2,
                    size=trace_size,
                    offset=C_sz_in_bytes,
                    start=0x1,
                    stop=0x0,
                    events=[0x4B, 0x22, 0x21, 0x25, 0x2D, 0x2C, 0x1A, 0x4F],
                )

            ipu_dma_memcpy_nd(
                metadata="out0", bd_id=0, mem=outTensor, sizes=[1, 1, 1, 64]
            )
            ipu_dma_memcpy_nd(
                metadata="in0", bd_id=1, mem=inTensor, sizes=[1, 1, 1, 64]
            )
            ipu_sync(column=0, row=0, direction=0, channel=0)


# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    my_first_aie_program() # Call design function within the mlir-aie context
    print(ctx.module) # Print the python-to-mlir conversion
