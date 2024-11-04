#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe | FileCheck %s
# CHECK: PASS!

import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

dtype = np.int16
repeat_count = 3
a_len = 2048
c_len = a_len * repeat_count


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            a_ty = np.ndarray[(a_len,), np.dtype[dtype]]
            c_ty = np.ndarray[(c_len,), np.dtype[dtype]]

            ShimTile = tile(0, 0)
            ComputeTile = tile(0, 2)
            fifo_a = object_fifo("fifo_a", ShimTile, ComputeTile, 2, a_ty)
            fifo_c = object_fifo("fifo_c", ComputeTile, ShimTile, 2, a_ty)

            # Core
            @core(ComputeTile)
            def core_body():
                for _ in range_(0, 0xFFFFFFFF):
                    for i in range_(repeat_count):
                        elem_c = fifo_c.acquire(ObjectFifoPort.Produce, 1)
                        elem_a = fifo_a.acquire(ObjectFifoPort.Consume, 1)
                        for i in range_(a_len):
                            elem_c[i] = elem_a[i]
                        fifo_a.release(ObjectFifoPort.Consume, 1)
                        fifo_c.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(a_ty, a_ty, c_ty)
            def sequence(A, _B, C):
                npu_dma_memcpy_nd(
                    metadata=fifo_a,
                    bd_id=1,
                    mem=A,
                    sizes=[repeat_count, 1, 1, a_len],
                    strides=[0, 0, 0, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_c,
                    bd_id=0,
                    mem=C,
                    sizes=[1, 1, 1, c_len],
                    strides=[0, 0, 0, 1],
                )
                dma_wait(fifo_c)

    print(ctx.module)


design()
