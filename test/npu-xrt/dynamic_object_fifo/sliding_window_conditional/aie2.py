#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/kernel.cc -o ./kernel.o
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host --dynamic-objFifos --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %run_on_npu1% ./test.exe

import numpy as np
import os

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import if_, else_

N = 100
n_rows = 10
dev = AIEDevice.npu1_1col
col = 0


def sliding_window():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            subtensor_ty = np.ndarray[(N // n_rows,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile, 2, subtensor_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, subtensor_ty)

            # AIE Core Function declarations
            add_10_i32 = external_func(
                "add_10_i32", inputs=[subtensor_ty, subtensor_ty, subtensor_ty]
            )

            # Set up compute tiles
            @core(ComputeTile, "kernel.o")
            def core_body():
                for i in range_(10):
                    elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                    with if_(i == 0) as if_op:
                        elemInPre = of_in.acquire(ObjectFifoPort.Consume, 1)
                        add_10_i32(elemInPre, elemInPre, elemOut)
                    with else_(if_op):
                        elemIn = of_in.acquire(ObjectFifoPort.Consume, 2)
                        add_10_i32(elemIn[0], elemIn[1], elemOut)
                        with if_(i == 9) as if_op1:
                            of_in.release(ObjectFifoPort.Consume, 2)
                        with else_(if_op1):
                            of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

            @runtime_sequence(tensor_ty, tensor_ty)
            def sequence(A, C):
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                dma_wait(of_out)

    print(ctx.module)


sliding_window()
