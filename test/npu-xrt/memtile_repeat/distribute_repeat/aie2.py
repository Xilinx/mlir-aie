#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %S/aie2.py 36 > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe -x final.xclbin -i insts.txt -k MLIR_AIE -l 36 | FileCheck %s
# CHECK: PASS!
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

dev = AIEDevice.npu1_1col
col = 0
N = 36

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

assert N % 2 == 0, "N must be even"
repeat_counter = 6
out_size = N * (repeat_counter + 1)


def distribute_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            dtype = np.dtype[np.int32]
            in_ty = np.ndarray[(N,), dtype]
            half_ty = np.ndarray[(N // 2,), dtype]
            out_ty = np.ndarray[(out_size,), dtype]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile2 = tile(col, 2)
            ComputeTile3 = tile(col, 3)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, in_ty)
            of_in2 = object_fifo("in2", MemTile, ComputeTile2, 2, half_ty)
            of_in3 = object_fifo("in3", MemTile, ComputeTile3, 2, half_ty)
            of_in2.set_memtile_repeat(repeat_counter)
            of_in3.set_memtile_repeat(repeat_counter)
            object_fifo_link(of_in, [of_in2, of_in3], [], [0, N // 2])

            of_out2 = object_fifo("out2", ComputeTile2, MemTile, 2, half_ty)
            of_out3 = object_fifo("out3", ComputeTile3, MemTile, 2, half_ty)
            of_out = object_fifo("out", MemTile, ShimTile, 1, out_ty)
            object_fifo_link([of_out2, of_out3], of_out, [0, out_size // 2], [])

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(N // 2):
                        elemOut[i] = elemIn[i] + 1
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce, 1)

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                for _ in range_(sys.maxsize):
                    elemOut = of_out3.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in3.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(N // 2):
                        elemOut[i] = elemIn[i] + 2
                    of_in3.release(ObjectFifoPort.Consume, 1)
                    of_out3.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(in_ty, in_ty, out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, out_size]
                )
                # of_out will only complete after of_in completes, so we just wait on of_out instead of both
                dma_wait(of_out)

    print(ctx.module)


distribute_repeat()
