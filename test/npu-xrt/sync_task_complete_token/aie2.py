#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test -std=c++11 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test | FileCheck %s
# CHECK: PASS!

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import arith
from aie.extras.dialects.ext.scf import _for as range_


dtype = T.i32
output_sz = 16

# This design produces `n_tiles` output tiles of size tile_sz.
# For each output tile, it reads the next contiguous 16 input tiles of size tile_sz, adds the values at each index together, and writes it to the output tile.
# In other words, this design produces:
#  output[i] = input[16*i] + input[16*i + 1] +  ... + input[16*i + 15]
# and the processing occurs in chuncks of tile_sz, i.e. one core call produces output[i], output[i+1], ... output[i+tile_sz-1]


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            memref_t = T.memref(1, dtype())

            # Tile declarations as tile[row][col]
            tiles = [[tile(col, row) for col in range(0, 4)] for row in range(0, 6)]
            # Shim tiles: tiles[0][0..3]
            # Mem tiles: tiles[1][0..3]
            # Cores: tiles[2..5][0..3]

            fifo_input = object_fifo(
                "fifo_input", tiles[0][0], tiles[2][0], 1, memref_t
            )
            fifo_output = object_fifo(
                "fifo_output", tiles[2][0], tiles[0][0], 1, memref_t
            )

            # Core
            @core(tiles[2][0])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_output = fifo_output.acquire(ObjectFifoPort.Produce, 1)
                    zero = arith.constant(T.i32(), 0)
                    memref.store(zero, elem_output, [0])
                    for _ in range_(16):
                        elem_input = fifo_input.acquire(ObjectFifoPort.Consume, 1)
                        a = memref.load(elem_output, [0])
                        b = memref.load(elem_input, [0])
                        c = a + b
                        memref.store(c, elem_output, [0])
                        fifo_input.release(ObjectFifoPort.Consume, 1)
                    fifo_output.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(memref_t, memref_t)
            def sequence(input, output):
                for i in range(output_sz):
                    # Configure and start, and wait for 16 BDs, each transferring the next contiguous input tile.
                    for j in range(16):
                        npu_dma_memcpy_nd(
                            metadata=fifo_input.sym_name.value,
                            bd_id=j,
                            mem=input,
                            offsets=[0, 0, 0, i * 16 + j],
                            sizes=[1, 1, 1, 1],
                            strides=[0, 0, 0, 1],
                            issue_token=True,
                        )
                        npu_dma_wait(fifo_input.sym_name.value)
                    # After transferring 16 input tiles, one output tile will be produced;
                    # issue a BD to transfer it back
                    npu_dma_memcpy_nd(
                        metadata=fifo_output.sym_name.value,
                        bd_id=0,
                        mem=output,
                        offsets=[0, 0, 0, i],
                        sizes=[1, 1, 1, 1],
                        strides=[0, 0, 0, 1],
                    )
                    npu_dma_wait(fifo_output.sym_name.value)

    print(ctx.module)


design()
