# section-2/section-2d/aie2.py -*- Python -*-
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
from aie.helpers.dialects.ext.scf import (
    _for as range_,
)  # scf (structured control flow) dialect

buffer_depth = 2
data_size = 48


# AI Engine structural design function
def mlir_aie_design():
    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # Device declaration - aie2 device xcvc1902
        @device(AIEDevice.npu1_1col)
        def device_body():
            data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

            # Tile(s) declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)

            # Data movement with object FIFOs

            # Input data movement
            of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile, buffer_depth, data_ty)
            object_fifo_link(of_in, of_in1)

            # Output data movement
            of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, data_ty)
            of_out1 = object_fifo("out1", ComputeTile, MemTile, buffer_depth, data_ty)
            object_fifo_link(of_out1, of_out)

            # Set up compute tiles
            @core(ComputeTile)
            def core_body():
                # Effective while(1)
                for _ in range_(0xFFFFFFFF):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(data_size):
                        elem_out[i] = elem_in[i] + 1
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)

    # Print the mlir conversion
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


# Call design function to generate mlir code to stdout
mlir_aie_design()
