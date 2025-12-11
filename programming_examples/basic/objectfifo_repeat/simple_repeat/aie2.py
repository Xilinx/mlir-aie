# test/npu-xrt/objectfifo_repeat/simple_repeat/aie2.py
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

N = 128
dev = AIEDevice.npu1_1col
col = 0
memtile_repeat_count = 4

if len(sys.argv) > 1:
    N = int(sys.argv[1])
data_out_size = N * memtile_repeat_count

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def simple_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tensor_ty = np.ndarray[(N,), np.dtype[np.uint32]]
            tensor_out_ty = np.ndarray[(64,), np.dtype[np.uint32]]
            tensor_check_ty = np.ndarray[(16,), np.dtype[np.uint32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, tensor_ty)
            of1 = object_fifo("mem_out", MemTile, ComputeTile, 1, tensor_check_ty)
            of2 = object_fifo("mem_in", ComputeTile, MemTile, 1, tensor_check_ty)
            of_out = object_fifo("out", MemTile, ShimTile, 1, tensor_check_ty)
            of1.set_repeat_count(memtile_repeat_count)
            # of2.set_repeat_count(memtile_repeat_count)
            # of_out.set_repeat_count(memtile_repeat_count)
            object_fifo_link(of_in, of1)
            object_fifo_link(of2, of_out)

            @core(ComputeTile)
            def core_body():
                for _ in range_(sys.maxsize):
                    memElemIn = of1.acquire(ObjectFifoPort.Consume, 1)
                    # Copy all 16 elements
                    for r in range_(1):
                        memElemOut = of2.acquire(ObjectFifoPort.Produce, 1)
                        for i in range_(16):
                            memElemOut[i] = memElemIn[i]
                        of2.release(ObjectFifoPort.Produce, 1)
                    of1.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty, tensor_out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=0,
                    mem=C,
                    sizes=[1, 1, 1, 64],
                )
                # of_out will only complete after of_in completes, so we just wait on of_out instead of both
                dma_wait(of_out)

    print(ctx.module)


simple_repeat()
