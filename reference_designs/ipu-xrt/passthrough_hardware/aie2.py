#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

N = 4096
N_in_bytes = N * 4

if len(sys.argv) == 2:
    N = int(sys.argv[1])

@constructAndPrintInModule
def my_passthrough():
    buffer_depth = 2

    @device(AIEDevice.ipu)
    def deviceBody():
        int32_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((1024,), int32_ty)

        # tile declarations
        ShimTile     = Tile(0, 0)
        ComputeTile2 = Tile(0, 2)

        # set up AIE-array data movement with Ordered Object Buffers
        OrderedObjectBuffer("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
        OrderedObjectBuffer("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)
        Link(["in"],["out"])
        
        memRef_tmp_ty = MemRefType.get((1,), int32_ty)
        @core(ComputeTile2)
        def coreBody():
            tmp = memref.AllocOp(memRef_tmp_ty, [], [])
            v0 = integerConstant(0, int32_ty)
            Store(v0, tmp, [0])
            

        memRef_mem_ty = MemRefType.get((N,), int32_ty)

        # to/from AIE-array data movement 
        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(A, B, C):
            IpuDmaMemcpyNd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            IpuDmaMemcpyNd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            IpuSync(column=0, row=0, direction=0, channel=0)
