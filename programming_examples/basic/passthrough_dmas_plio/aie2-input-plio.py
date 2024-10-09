# passthrough_dmas_plio/aie2-output-plio.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

N = 1024
line_size = 1024

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0
dev = AIEDevice.xcvc1902


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile1 = tile(30, 0)
            ShimTile2 = tile(26, 0)
            ComputeTile2 = tile(30, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile1, ComputeTile2, 2, line_ty, plio=True)
            of_out = object_fifo("out", ComputeTile2, ShimTile2, 2, line_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
                # of_out will only complete after of_in completes, so we just wait on of_out instead of both
                dma_wait(of_out)

    print(ctx.module)


my_passthrough()
