# section-2/section-2d/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.dialects.scf import *  # scf (strcutred control flow) dialect
from aie.extras.dialects.ext import memref, arith  # memref and arithmatic dialects

buffer_depth = 2
data_size = 48


# AI Engine structural design function
def mlir_aie_design():
    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # Device declaration - aie2 device xcvc1902
        @device(AIEDevice.xcvc1902)
        def device_body():
            memRef_48_ty = T.memref(data_size, T.i32())

            # Tile(s) declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)

            # Data movement with object FIFOs

            # Input data movement

            of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, memRef_48_ty)
            of_in1 = object_fifo(
                "in1", MemTile, ComputeTile, buffer_depth, memRef_48_ty
            )
            object_fifo_link(of_in, of_in1)

            # Output data movement

            of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, memRef_48_ty)
            of_out1 = object_fifo(
                "out1", ComputeTile, MemTile, buffer_depth, memRef_48_ty
            )
            object_fifo_link(of_out1, of_out)

            # Set up compute tiles
            @core(ComputeTile)
            def core_body():
                # Effective while(1)
                for _ in for_(0xFFFFFFFF):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(data_size):
                        v0 = memref.load(elem_in, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)
                    yield_([])

    # Print the mlir conversion
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


# Call design function to generate mlir code to stdout
mlir_aie_design()
