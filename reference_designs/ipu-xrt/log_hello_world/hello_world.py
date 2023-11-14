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
from aie.passmanager import PassManager

def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        pm = PassManager("builtin.module")
        pm.add("canonicalize")
        pm.run(module.operation)
        print(module)

@constructAndPrintInModule
def printf():
    N = 512

    @device(AIEDevice.ipu)
    def deviceBody():
        int32_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((N,), int32_ty)

        kernel = privateFunc("kernel", inputs=[memRef_ty, memRef_ty, memRef_ty])

        ShimTile = Tile(0, 0)
        ComputeTile = Tile(0, 2)

        OrderedObjectBuffer("inOF", ShimTile, ComputeTile, 2, memRef_ty)
        OrderedObjectBuffer("outOF", ComputeTile, ShimTile, 2, memRef_ty)
        OrderedObjectBuffer("logoutOF", ComputeTile, ShimTile, 2, memRef_ty)

        @core(ComputeTile, "kernel.o")
        def coreBody():
            elemOut = Acquire(
                "outOF", ObjectFifoPort.Produce, 1, memRef_ty
            ).acquiredElem()
            elemIn = Acquire(
                "inOF", ObjectFifoPort.Consume, 1, memRef_ty
            ).acquiredElem()
            elemLogout = Acquire(
                "logoutOF", ObjectFifoPort.Produce, 1, memRef_ty
            ).acquiredElem()
            Call(kernel, [elemIn, elemOut, elemLogout])
            Release(ObjectFifoPort.Consume, "inOF", 1)
            Release(ObjectFifoPort.Produce, "outOF", 1)
            Release(ObjectFifoPort.Produce, "logoutOF", 1)

        @FuncOp.from_py_func(memRef_ty, memRef_ty, memRef_ty)
        def sequence(in_mem, out_mem, logout):
            IpuDmaMemcpyNd(metadata="outOF", bd_id=0, mem=out_mem, lengths=[1, 1, 1, N])
            IpuDmaMemcpyNd(metadata="inOF", bd_id=1, mem=in_mem, lengths=[1, 1, 1, N])
            IpuDmaMemcpyNd(metadata="logoutOF", bd_id=2, mem=logout, lengths=[1, 1, 1, N])
            IpuSync(column=0, row=0, direction=0, channel=0)