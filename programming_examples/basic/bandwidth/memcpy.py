# memcpy.py -*- Python -*-
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
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

N = 32768
dev = AIEDevice.npu1_4col
cols = 4
chs = 2
line_size = 1024

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0

assert (N % line_size) % cols % chs == 0

M = N // cols // chs


def my_memcpy():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTiles = [tile(i, 0) for i in range(cols)]
            MemTiles = [tile(i, 1) for i in range(cols)]
            of_ins = [
                object_fifo(f"in{i}_{j}", ShimTiles[i], MemTiles[i], 2, line_ty)
                for i in range(cols)
                for j in range(chs)
            ]
            of_outs = [
                object_fifo(f"out{i}_{j}", MemTiles[i], ShimTiles[i], 2, line_ty)
                for i in range(cols)
                for j in range(chs)
            ]
            for col in range(cols):
                for j in range(chs):
                    object_fifo_link(of_ins[col * chs + j], of_outs[col * chs + j])

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                for i in range(cols):
                    for j in range(chs):
                        npu_dma_memcpy_nd(
                            metadata=of_outs[i * chs + j],
                            bd_id=j * chs + 0,
                            mem=C,
                            sizes=[1, 1, 1, M],
                            offsets=[1, 1, 1, M * i * chs + M * j],
                        )
                for i in range(cols):
                    for j in range(chs):
                        npu_dma_memcpy_nd(
                            metadata=of_ins[i * chs + j],
                            bd_id=j * chs + 1,
                            mem=A,
                            sizes=[1, 1, 1, M],
                            offsets=[1, 1, 1, M * i * chs + M * j],
                        )
                dma_wait(*of_outs)

    print(ctx.module)


my_memcpy()
