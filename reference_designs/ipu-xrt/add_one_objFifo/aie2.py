#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import aie
from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *
from aie.dialects.aiex import *

@constructAndPrintInModule
def my_add_one_objFifo():
    @device(AIEDevice.ipu)
    def deviceBody():
        int32_ty     = IntegerType.get_signless(32)
        memRef_16_ty = MemRefType.get((16,), int32_ty)
        memRef_8_ty  = MemRefType.get((8,), int32_ty)

        ShimTile     = Tile(0, 0)
        MemTile      = Tile(0, 1)
        ComputeTile2 = Tile(0, 2)

        OrderedObjectBuffer("in0", ShimTile, MemTile, 2, memRef_16_ty)
        OrderedObjectBuffer("in1", MemTile, ComputeTile2, 2, memRef_8_ty)
        Link(["in0"], ["in1"])
        OrderedObjectBuffer("out0", MemTile, ShimTile, 2, memRef_8_ty)
        OrderedObjectBuffer("out1", ComputeTile2, MemTile, 2, memRef_16_ty)
        Link(["out1"], ["out0"])

        @core(ComputeTile2)
        def coreBody():
            # Effective while(1)
            @forLoop(lowerBound=0, upperBound=8, step=1)
            def loopTile():
                elemIn = Acquire(
                    ObjectFifoPort.Consume, "in1", 1, memRef_8_ty
                ).acquiredElem()
                elemOut = Acquire(
                    ObjectFifoPort.Produce, "out1", 1, memRef_8_ty
                ).acquiredElem()
                # @forLoop(lowerBound=0, upperBound=8, step=1)
                #   load elemIn[idx]
                #   add 1
                #   store elemOut[idx]
                Release(ObjectFifoPort.Consume, "in1", 1)
                Release(ObjectFifoPort.Produce, "out1", 1)

        memRef_64_ty = MemRefType.get((64,), int32_ty)
        memRef_32_ty = MemRefType.get((64,), int32_ty)

        @FuncOp.from_py_func(memRef_64_ty, memRef_32_ty, memRef_64_ty)
        def sequence(inTensor, notUsed, outTensor):
            IpuDmaMemcpyNd(metadata="out0", bd_id=0, mem=outTensor, lengths=[1, 1, 1, 64])
            IpuDmaMemcpyNd(metadata="in0", bd_id=1, mem=inTensor, lengths=[1, 1, 1, 64])
            IpuSync(column=0, row=0, direction=0, channel=0)
