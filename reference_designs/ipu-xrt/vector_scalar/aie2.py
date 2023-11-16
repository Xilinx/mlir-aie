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

@constructAndPrintInModule
def my_vector_scalar():
    N = 4096
    n = 1024
    N_div_n = N // n
    N_in_bytes = N * 4

    buffer_depth = 2

    @device(AIEDevice.ipu)
    def deviceBody():
        int32_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((n,), int32_ty)

        scale_int32 = privateFunc("scale_int32", inputs=[memRef_ty, memRef_ty])

        # tile declarations
        ShimTile     = Tile(0, 0)
        ComputeTile2 = Tile(0, 2)

        # set up AIE-array data movement with Ordered Object Buffers
        OrderedObjectBuffer("in", ShimTile, ComputeTile2, buffer_depth, memRef_ty)
        OrderedObjectBuffer("out", ComputeTile2, ShimTile, buffer_depth, memRef_ty)

        # set up compute tiles
        @core(ComputeTile2, "scale.o")
        def coreBody():
            # Effective while(1)
            @forLoop(lowerBound=0, upperBound=sys.maxsize, step=1)
            def loopReps():
                # Number of sub-vector "tile" iterations
                @forLoop(lowerBound=0, upperBound=N_div_n, step=1)
                def loopTile():
                    elemOut = Acquire(
                        ObjectFifoPort.Produce, "out", 1, memRef_ty
                    ).acquiredElem()
                    elemIn = Acquire(
                        ObjectFifoPort.Consume, "in", 1, memRef_ty
                    ).acquiredElem()
                    Call(scale_int32, [elemIn, elemOut])
                    Release(ObjectFifoPort.Consume, "in", 1)
                    Release(ObjectFifoPort.Produce, "out", 1)

        memRef_mem_ty = MemRefType.get((N,), int32_ty)

        # to/from AIE-array data movement 
        @FuncOp.from_py_func(memRef_mem_ty, memRef_mem_ty, memRef_mem_ty)
        def sequence(A, B, C):
            IpuDmaMemcpyNd(metadata="out", bd_id=0, mem=C, lengths=[1, 1, 1, N])
            IpuDmaMemcpyNd(metadata="in", bd_id=1, mem=A, lengths=[1, 1, 1, N])
            IpuSync(column=0, row=0, direction=0, channel=0)
