#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/kernel.cc -o ./kernel.o
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test -std=c++11 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test | FileCheck %s
# CHECK: PASS!
import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

dtype = np.int16
a_len = 8
b_len = 12
c_offset = 2
c_len = a_len + b_len


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            a_ty = np.ndarray[(a_len,), np.dtype[dtype]]
            b_ty = np.ndarray[(b_len,), np.dtype[dtype]]
            c_ty = np.ndarray[(c_len,), np.dtype[dtype]]

            concat_func = external_func(
                "concat",
                inputs=[a_ty, b_ty, c_ty, np.int32, np.int32, np.int32],
            )

            # Tile declarations as tile[row][col]
            tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
            # Shim tiles: tiles[0][0..3]
            # Mem tiles: tiles[1][0..3]
            # Cores: tiles[2..5][0..3]

            fifo_a = object_fifo("fifo_a", tiles[0][0], tiles[2][0], 2, a_ty)
            fifo_b = object_fifo("fifo_b", tiles[0][0], tiles[2][0], 2, b_ty)
            fifo_c = object_fifo("fifo_c", tiles[2][0], tiles[0][0], 2, c_ty)

            # Core
            @core(tiles[2][0], "kernel.o")
            def core_body():
                for _ in range_(0, 0xFFFFFFFF):
                    elem_c = fifo_c.acquire(ObjectFifoPort.Produce, 1)
                    elem_a = fifo_a.acquire(ObjectFifoPort.Consume, 1)
                    elem_b = fifo_b.acquire(ObjectFifoPort.Consume, 1)
                    concat_func(
                        elem_a,
                        elem_b,
                        elem_c,
                        a_len,
                        b_len,
                        c_len,
                    )
                    fifo_a.release(ObjectFifoPort.Consume, 1)
                    fifo_b.release(ObjectFifoPort.Consume, 1)
                    fifo_c.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(a_ty, b_ty, c_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=fifo_a,
                    bd_id=1,
                    mem=A,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, a_len // 4, 2, 2],
                    strides=[0, 2, a_len // 2, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_b,
                    bd_id=1,
                    mem=B,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 2, b_len // 4, 2],
                    strides=[0, 2, 4, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_c,
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, c_offset],
                    sizes=[1, 1, 1, c_len],
                    strides=[0, 0, 0, 1],
                )
                dma_wait(fifo_c)

    print(ctx.module)


design()
