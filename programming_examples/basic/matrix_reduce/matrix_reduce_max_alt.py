#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir ctx wrapper

from aie.dialects.aiex import *  # extended mlir-aie dialect definitions
from aie.helpers.dialects.ext.scf import (
    _for as range_,
)  # scf (structured control flow) dialect
from aie.helpers.util import np_ndarray_type_get_shape


# AI Engine structural design function
def matrix_reduce_max():

    # Device declaration - aie2 device NPU (aka Ryzen AI)
    @device(AIEDevice.npu1_1col)
    def device_body():

        N = 512
        C = 1  # FIXME breaks if C > 1
        n_cores = C

        # Define tensor types
        in_ty = np.ndarray[
            (
                N,
                N,
            ),
            np.dtype[np.int32],
        ]
        out_ty = np.ndarray[(N,), np.dtype[np.int32]]

        # Define tensor tile types
        in_tile_ty = np.ndarray[
            (
                C,
                N,
            ),
            np.dtype[np.int32],
        ]
        out_tile_ty = np.ndarray[(C,), np.dtype[np.int32]]

        # Define worker tensor tile types
        in_worker_ty = np.ndarray[(N,), np.dtype[np.int32]]
        out_worker_ty = np.ndarray[(1,), np.dtype[np.int32]]

        # AIE Core Function declarations
        reduce_add_vector = external_func(
            "reduce_max_vector", inputs=[in_worker_ty, out_worker_ty, np.int32]
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", ShimTile, MemTile, 2, in_tile_ty)
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"memA{i}", MemTile, cores[i], 2, in_worker_ty)
            )
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(in_tile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(inA, inA_fifos, [], of_offsets)

        # Output C
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, 2, out_worker_ty)
            )
        outC = object_fifo("outC", MemTile, ShimTile, 2, out_tile_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(out_tile_ty)) // n_cores) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifos, outC, of_offsets, [])

        # Compute tile bodies
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "reduce_max.cc.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)

                    reduce_add_vector(elem_in_a, elem_out, N)

                    inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                    outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(in_ty, out_ty)
        def sequence(A, C):
            in_task = shim_dma_single_bd_task(
                inA, A, sizes=[1, 1, 1, N * N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


with mlir_mod_ctx() as ctx:
    matrix_reduce_max()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
