#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

N = 4096
n = 1024
N_div_n = N // n
N_in_bytes = N * 4

buffer_depth = 2

@constructAndPrintInModule
def my_vector_scalar():
    @device("ipu")
    def deviceBody():
        int32_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((n,), int32_ty)

        scale_int32 = privateFunc("scale_int32", inputs = [memRef_ty, memRef_ty])

        S = Tile(0, 0)
        T = Tile(0, 2)

        OrderedObjectBuffer("in", S, T, buffer_depth, memRef_ty)
        OrderedObjectBuffer("out", T, S, buffer_depth, memRef_ty)

        @core(T, "scale.o")
        def coreBody():
            # Effective while(1)
            @forLoop(lowerBound = 0, upperBound = 0XFFFFFFFF, step = 1)
            def loopReps():
                # Number of sub-vector "tile" iterations
                @forLoop(lowerBound = 0, upperBound = N_div_n, step = 1)
                def loopTile():
                    elemOut = Acquire("out", "Produce", 1, memRef_ty).acquiredElem() 
                    elemIn = Acquire("in", "Consume", 1, memRef_ty).acquiredElem()
                    call(scale_int32, [elemIn, elemOut])
                    Release("in", "Consume", 1)
                    Release("out", "Produce", 1)


        memRef_mem_ty =  MemRefType.get((N,), int32_ty)
        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(A, B, C):
            IpuDmaMemcpyNd(metadata = "out", bd_id = 0, mem = C, lengths = [1, 1, 1, N])
            IpuDmaMemcpyNd(metadata = "in", bd_id = 1, mem = A, lengths = [1, 1, 1, N])
            IpuSync(column = 0, row = 0, direction = 0, channel = 0)
