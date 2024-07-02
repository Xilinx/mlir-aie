#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai
#
# RUN: make -f %S/Makefile clean
# RUN: env SRC=%S AIETOOLS=%aietools AIECC=$(which aiecc.py) XRT_FLAGS="%xrt_flags" make -f %S/Makefile test
# RUN: make -f %S/Makefile run | FileCheck %s
# CHECK: PASS!

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


matrix_rows = 7
matrix_cols = 19
matrix_size = matrix_rows * matrix_cols


def memref_sz(m: MemRefType):
    sz = 1
    for s in m.shape:
        sz *= s
    return sz


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            matrix_memref = T.memref(matrix_size, T.i32())

            passthrough_func = external_func(
                "passthrough", inputs=[matrix_memref, matrix_memref, T.i32()]
            )

            # Tile declarations as tile[row][col]
            tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
            # Shim tiles: tiles[0][0..3]
            # Mem tiles: tiles[1][0..3]
            # Cores: tiles[2..5][0..3]

            fifo_in = object_fifo("fifo_in", tiles[0][0], tiles[2][0], 2, matrix_memref)
            fifo_out = object_fifo(
                "fifo_out", tiles[2][0], tiles[0][0], 2, matrix_memref
            )

            # Core
            @core(tiles[2][0], "kernel.o")
            def core_body():
                for _ in for_(0, 0xFFFFFFFF):
                    elem_in = fifo_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = fifo_out.acquire(ObjectFifoPort.Produce, 1)
                    call(passthrough_func, [elem_in, elem_out, matrix_size])
                    fifo_in.release(ObjectFifoPort.Consume, 1)
                    fifo_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            @FuncOp.from_py_func(matrix_memref, matrix_memref)
            def sequence(inp, out):
                npu_dma_memcpy_nd(
                    metadata=fifo_in.sym_name.value,
                    bd_id=1,
                    mem=inp,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, matrix_cols, matrix_rows],
                    strides=[0, 0, 1, matrix_cols],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_out.sym_name.value,
                    bd_id=0,
                    mem=out,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, matrix_rows * matrix_cols],
                    strides=[0, 0, 0, 1],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


design()
