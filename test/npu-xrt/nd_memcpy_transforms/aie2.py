#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


dtype = T.i16
a_len = 8
b_len = 12
c_offset = 2
c_len = a_len + b_len 


def memref_sz(m : MemRefType):
    sz = 1
    for s in m.shape:
        sz *= s
    return sz


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            memref_a = T.memref(a_len, dtype())
            memref_b = T.memref(b_len, dtype())
            memref_c = T.memref(c_len, dtype())

            concat_func = external_func("concat", inputs=[memref_a, memref_b, memref_c, T.i32(), T.i32(), T.i32()])

            # Tile declarations as tile[row][col]
            tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
            # Shim tiles: tiles[0][0..3]
            # Mem tiles: tiles[1][0..3]
            # Cores: tiles[2..5][0..3]

            fifo_a = object_fifo("fifo_a", tiles[0][0], tiles[2][0], 2, memref_a)
            fifo_b = object_fifo("fifo_b", tiles[0][0], tiles[2][0], 2, memref_b)
            fifo_c = object_fifo("fifo_c", tiles[2][0], tiles[0][0], 2, memref_c)

            # Core
            @core(tiles[2][0], "kernel.o")
            def core_body():
                for _ in for_(0, 0xFFFFFFFF):
                    elem_c = fifo_c.acquire(ObjectFifoPort.Produce, 1)
                    elem_a = fifo_a.acquire(ObjectFifoPort.Consume, 1)
                    elem_b = fifo_b.acquire(ObjectFifoPort.Consume, 1)
                    call(concat_func, [elem_a, elem_b, elem_c, memref_sz(memref_a), memref_sz(memref_b), memref_sz(memref_c)])
                    fifo_a.release(ObjectFifoPort.Consume, 1)
                    fifo_b.release(ObjectFifoPort.Consume, 1)
                    fifo_c.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            @FuncOp.from_py_func(memref_a, memref_b, memref_c)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=fifo_a.sym_name.value, 
                    bd_id=1, 
                    mem=A,
                    offsets=[0,0,0,0],
                    sizes=[1, a_len//4, 2, 2],
                    strides=[0, 2, a_len//2]
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_b.sym_name.value, 
                    bd_id=1, 
                    mem=B,
                    offsets=[0,0,0,0],
                    sizes=[1, 2, b_len//4, 2],
                    strides=[0, 2, 4]
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_c.sym_name.value, 
                    bd_id=0, 
                    mem=C,
                    offsets=[0, 0, 0, c_offset],
                    sizes=[1, 1, 1, c_len],
                    strides=[0, 0, 0]
                )
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


design()