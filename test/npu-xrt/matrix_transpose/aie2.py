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
# RUN: clang %S/test.cpp -o test -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test | FileCheck %s
# CHECK: PASS!
import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

matrix_rows = 7
matrix_cols = 19
matrix_size = matrix_rows * matrix_cols


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            matrix_ty = np.ndarray[(matrix_size,), np.dtype[np.int32]]

            passthrough_func = external_func(
                "passthrough", inputs=[matrix_ty, matrix_ty, np.int32]
            )

            # Tile declarations as tile[row][col]
            tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
            # Shim tiles: tiles[0][0..3]
            # Mem tiles: tiles[1][0..3]
            # Cores: tiles[2..5][0..3]

            fifo_in = object_fifo("fifo_in", tiles[0][0], tiles[2][0], 2, matrix_ty)
            fifo_out = object_fifo("fifo_out", tiles[2][0], tiles[0][0], 2, matrix_ty)

            # Core
            @core(tiles[2][0], "kernel.o")
            def core_body():
                for _ in range_(0, 0xFFFFFFFF):
                    elem_in = fifo_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = fifo_out.acquire(ObjectFifoPort.Produce, 1)
                    passthrough_func(elem_in, elem_out, matrix_size)
                    fifo_in.release(ObjectFifoPort.Consume, 1)
                    fifo_out.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(matrix_ty, matrix_ty)
            def sequence(inp, out):
                npu_dma_memcpy_nd(
                    metadata=fifo_in,
                    bd_id=1,
                    mem=inp,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, matrix_cols, matrix_rows],
                    strides=[0, 0, 1, matrix_cols],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_out,
                    bd_id=0,
                    mem=out,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, matrix_rows * matrix_cols],
                    strides=[0, 0, 0, 1],
                )
                dma_wait(fifo_out)

    print(ctx.module)


design()
