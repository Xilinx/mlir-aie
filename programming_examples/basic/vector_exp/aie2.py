# vector_exp/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.extras.dialects.ext.scf import (
    _for as range_,
)  # scf (structured control flow) dialect


# AI Engine structural design function
def my_eltwise_exp():

    N = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores
    buffer_depth = 2

    # Device declaration - aie2 device NPU (aka Ryzen AI)
    @device(AIEDevice.npu1_1col)
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

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, buffer_depth, memRef_A_MT_ty)
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], buffer_depth, memRef_A_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(memRef_A_MT_ty.shape) // n_cores) * i for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inA, inA_fifos, [], of_offsets)

        # Output C
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, memRef_C_ty)
            )
        outC = object_fifo("outC", MemTile, ShimTile, buffer_depth, memRef_C_MT_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(memRef_C_MT_ty.shape) // n_cores) * i for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifos, outC, of_offsets, [])

        # Compute tile bodies
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "kernels.a")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)

                        exp_bf16_1024(elem_in_a, elem_out)

                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.bf16())

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, C):
            npu_dma_memcpy_nd(
                metadata=inA, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
            )
            npu_dma_memcpy_nd(metadata=outC, bd_id=0, mem=C, sizes=[1, 1, 1, N])
            dma_wait(inA, outC)


with mlir_mod_ctx() as ctx:
    my_eltwise_exp()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
