#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 4096
if len(sys.argv) == 2:
    N = int(sys.argv[1])

n_columns = 4 
n_fifos = 2

assert(N%1024 == 0)
assert(N%n_columns == 0)
assert(n_columns * n_fifos <= 16)  # Or we'll run out of BDs

def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            memRef_ty = T.memref(1024, T.i32())

            # Tile declarations
            shim_tiles = []
            compute_tiles = []
            for i in range(n_columns):
                shim_tile = tile(i, 0)
                compute_tile = tile(i, 2)
                shim_tiles.append(shim_tile)
                compute_tiles.append(compute_tile)

            # AIE-array data movement with object fifos
            for i, (shim, compute) in enumerate(zip(shim_tiles, compute_tiles)):
                for j in range(n_fifos):
                    of_in  = object_fifo(f"in_{i}_{j}", shim, compute, 2, memRef_ty)
                    of_out = object_fifo(f"out_{i}_{j}", compute, shim, 2, memRef_ty)
                    object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tiles
            for i in range(n_columns):
                @core(compute_tiles[i])
                def core_body():
                    tmp = memref.alloc(1, T.i32())
                    v0 = arith.constant(0, T.i32())
                    memref.store(v0, tmp, [0])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                tile_N = N//n_columns//n_fifos
                for i in range(n_columns):
                    for j in range(n_fifos):
                        ipu_dma_memcpy_nd(metadata=f"out_{i}_{j}", bd_id=2*n_fifos*i+j, mem=C,  sizes=[1, 1, 1, tile_N], offsets=[0, 0, 0, i*n_fifos*tile_N + j*tile_N])
                        ipu_dma_memcpy_nd(metadata=f"in_{i}_{j}", bd_id=2*n_fifos*i+1+j, mem=A, sizes=[1, 1, 1, tile_N], offsets=[0, 0, 0, i*n_fifos*tile_N + j*tile_N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


my_passthrough()
