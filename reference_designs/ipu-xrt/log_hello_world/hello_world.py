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
# from aie.passmanager import PassManager
from aie.util import mlir_mod_ctx

# def constructAndPrintInModule(f):
#     with Context() as ctx, Location.unknown():
#         aie.dialects.aie.register_dialect(ctx)
#         module = Module.create()
#         with InsertionPoint(module.body):
#             f()
#         pm = PassManager("builtin.module")
#         pm.add("canonicalize")
#         pm.run(module.operation)
#         print(module)

# @constructAndPrintInModule
def printf():
    N = 512

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.ipu)
        def device_body():
            # int32_ty = IntegerType.get_signless(32)
            # memRef_ty = MemRefType.get((N,), int32_ty)
            memRef_ty = TypeAttr.get(ObjectFifoType.get(T.memref(N, T.i32())))

            kernel = external_func("kernel", inputs=[T.memref(N, T.i32()), 
                                   T.memref(N, T.i32()), T.memref(N, T.i32())])

            ShimTile     = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            objectfifo("inOF", ShimTile, [ComputeTile2], 2, memRef_ty, [], [])
            objectfifo("outOF", ComputeTile2, [ShimTile], 2, memRef_ty, [], [])
            objectfifo("logoutOF", ComputeTile2, [ShimTile], 2, memRef_ty, [], [])

            @core(ComputeTile2, "kernel.o")
            def core_body():
                elemOut = acquire(
                    ObjectFifoPort.Produce, "outOF", 1, T.memref(N, T.i32())
                ).acquired_elem()
                elemIn = acquire(
                    ObjectFifoPort.Consume, "inOF", 1, T.memref(N, T.i32())
                ).acquired_elem()
                elemLogout = acquire(
                    ObjectFifoPort.Produce, "logoutOF", 1, T.memref(N, T.i32())
                ).acquired_elem()
                Call(kernel, [elemIn, elemOut, elemLogout])
                objectfifo_release(ObjectFifoPort.Consume, "inOF", 1)
                objectfifo_release(ObjectFifoPort.Produce, "outOF", 1)
                objectfifo_release(ObjectFifoPort.Produce, "logoutOF", 1)

            @FuncOp.from_py_func(
                T.memref(N, T.i32()), T.memref(N, T.i32()), T.memref(N, T.i32())
            )
            def sequence(in_mem, out_mem, logout):
                ipu_dma_memcpy_nd(metadata="outOF", bd_id=0, mem=out_mem, lengths=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="inOF", bd_id=1, mem=in_mem, lengths=[1, 1, 1, N])
                ipu_dma_memcpy_nd(metadata="logoutOF", bd_id=2, mem=logout, lengths=[1, 1, 1, N])
                ipu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)

printf()
