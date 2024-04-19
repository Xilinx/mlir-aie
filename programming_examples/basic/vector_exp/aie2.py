#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.dialects.scf import *  # scf (strcutred control flow) dialect
from aie.extras.dialects.ext import memref, arith  # memref and arithmatic dialects


# AI Engine structural design function
def my_eltwise_exp():

    word_size_in = 2
    N = 65536
    N_in_bytes = N * word_size_in

    A_sz_in_i32s = N_in_bytes // 4
    C_sz_in_i32s = N_in_bytes // 4

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores
    buffer_depth = 2

    # Device declaration - aie2 device NPU (aka Ryzen AI)
    @device(AIEDevice.npu)
    def device_body():

        memRef_ty = T.memref(n, T.bf16())

        # Type used in the tile memory
        memRef_A_ty = T.memref(n, T.bf16())
        memRef_C_ty = T.memref(n, T.bf16())

        # Type used in the memory tile which aggregates across the 4 cores
        memRef_A_MT_ty = T.memref(n * n_cores, T.bf16())
        memRef_C_MT_ty = T.memref(n * n_cores, T.bf16())

        # AIE Core Function declarations

        exp_bf16_1024 = external_func("exp_bf16_1024", inputs=[memRef_ty, memRef_ty])

        # Tile declarations
        ShimTile = tile(0, 0)

        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifo_names = [f"memA{i}" for i in range(n_cores)]
        outC_fifo_names = [f"memC{i}" for i in range(n_cores)]

        inA_fifos = {}
        outC_fifos = {}

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, memRef_A_MT_ty)
        for i in range(n_cores):
            inA_fifos[inA_fifo_names[i]] = object_fifo(
                inA_fifo_names[i], MemTile, cores[i], buffer_depth, memRef_A_ty
            )
        object_fifo_link(inA, inA_fifo_names)

        # Output C
        for i in range(n_cores):
            outC_fifos[outC_fifo_names[i]] = object_fifo(
                outC_fifo_names[i], cores[i], MemTile, buffer_depth, memRef_C_ty
            )
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, memRef_C_MT_ty)
        object_fifo_link(outC_fifo_names[0:n_cores], outC)

        # Compute tile bodies
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "kernels.a")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles):
                        elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                            ObjectFifoPort.Consume, 1
                        )

                        call(exp_bf16_1024, [elem_in_a, elem_out])

                        inA_fifos[inA_fifo_names[i]].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[outC_fifo_names[i]].release(
                            ObjectFifoPort.Produce, 1
                        )
                        yield_([])
                    yield_([])

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.i32())

        @FuncOp.from_py_func(tensor_ty, tensor_ty)
        def sequence(A, C):
            npu_dma_memcpy_nd(
                metadata="outC", bd_id=0, mem=C, sizes=[1, 1, 1, C_sz_in_i32s]
            )
            npu_dma_memcpy_nd(
                metadata="inA", bd_id=1, mem=A, sizes=[1, 1, 1, A_sz_in_i32s]
            )
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_eltwise_exp()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
