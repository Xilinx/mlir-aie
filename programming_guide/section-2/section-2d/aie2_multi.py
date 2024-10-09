# section-2/section-2d/aie2_multi.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.extras.dialects.ext.scf import (
    _for as range_,
)  # scf (structured control flow) dialect

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
            tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
            data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

            # Tile(s) declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTiles = [tile(0, 2 + i) for i in range(n_cores)]

            # Data movement with object FIFOs

            # Input data movement
            inX_fifos = []

            of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
            for i in range(n_cores):
                inX_fifos.append(
                    object_fifo(
                        f"in{i}",
                        MemTile,
                        ComputeTiles[i],
                        buffer_depth,
                        tile_ty,
                    )
                )
            if n_cores > 1:
                of_offsets = [tile_size * i for i in range(n_cores)]
            else:
                of_offsets = []
            object_fifo_link(of_in, inX_fifos, [], of_offsets)

            # Output data movement
            outX_fifos = []

            of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, data_ty)
            for i in range(n_cores):
                outX_fifos.append(
                    object_fifo(
                        f"out{i}",
                        ComputeTiles[i],
                        MemTile,
                        buffer_depth,
                        tile_ty,
                    )
                )
            object_fifo_link(outX_fifos, of_out, of_offsets, [])

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(ComputeTiles[i])
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_in = inX_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_out = outX_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        for j in range_(tile_size):
                            elem_out[j] = elem_in[j] + 1
                        inX_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outX_fifos[i].release(ObjectFifoPort.Produce, 1)

    # Print the mlir conversion
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


# Call design function to generate mlir code to stdout
mlir_aie_design()
