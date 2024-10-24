# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext.scf import _for as range_

N = 56
dev = AIEDevice.npu1_1col
col = 0

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(25, T.i32())
            memRef_ty2 = T.memref(56, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, memRef_ty)
            of_out = object_fifo(
                "out",
                MemTile,
                ShimTile,
                1,
                memRef_ty2,
                dimensionsToStream=[(5, 5), (5, 5)],
                padDimensions=[(2, 0), (3, 0)],
            )
            object_fifo_link(of_in, of_out)

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
                )
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
                dma_wait(of_in, of_out)

    print(ctx.module)


my_passthrough()
