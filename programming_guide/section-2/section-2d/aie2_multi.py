# section-2/section-2d/aie2_multi.py -*- Python -*-
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

n_cores = 3
buffer_depth = 2
data_size = 48
tile_size = data_size // 3


# AI Engine structural design function
def mlir_aie_design():
    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # Device declaration - aie2 device xcvc1902
        @device(AIEDevice.xcvc1902)
        def device_body():
            memRef_16_ty = T.memref(16, T.i32())
            memRef_48_ty = T.memref(48, T.i32())

            # Tile(s) declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTiles = [tile(0, 2 + i) for i in range(n_cores)]

            # Data movement with object FIFOs

            # Input data movement

            inX_fifo_names = [
                f"in{i}" for i in range(n_cores)
            ]  # list of input object FIFO names
            inX_fifos = {}  # map name to its object FIFO

            of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, memRef_48_ty)
            for i in range(n_cores):
                inX_fifos[inX_fifo_names[i]] = object_fifo(
                    inX_fifo_names[i],
                    MemTile,
                    ComputeTiles[i],
                    buffer_depth,
                    memRef_16_ty,
                )
            object_fifo_link(of_in, inX_fifo_names[0:n_cores])

            # Output data movement

            outX_fifo_names = [
                f"out{i}" for i in range(n_cores)
            ]  # list of output object FIFO names
            outX_fifos = {}  # map name to its object FIFO

            of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, memRef_48_ty)
            for i in range(n_cores):
                outX_fifos[outX_fifo_names[i]] = object_fifo(
                    outX_fifo_names[i],
                    ComputeTiles[i],
                    MemTile,
                    buffer_depth,
                    memRef_16_ty,
                )
            object_fifo_link(outX_fifo_names[0:n_cores], of_out)

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(ComputeTiles[i])
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        elem_in = inX_fifos[inX_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        elem_out = outX_fifos[outX_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        for j in for_(tile_size):
                            v0 = memref.load(elem_in, [j])
                            v1 = arith.addi(v0, arith.constant(1, T.i32()))
                            memref.store(v1, elem_out, [j])
                            yield_([])
                        inX_fifos[inX_fifo_names[i]].release(ObjectFifoPort.Consume, 1)
                        outX_fifos[outX_fifo_names[i]].release(
                            ObjectFifoPort.Produce, 1
                        )
                        yield_([])

    # Print the mlir conversion
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


# Call design function to generate mlir code to stdout
mlir_aie_design()
