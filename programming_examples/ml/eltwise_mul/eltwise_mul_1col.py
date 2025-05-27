#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_eltwise_mul(dev):
    word_size_in = 2
    N = 65536
    N_in_bytes = N * word_size_in

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores
    N_per_shimtile = N // n_cores
    buffer_depth = 2

    @device(dev)
    def device_body():
        tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the tile memory
        A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        B_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        C_ty = np.ndarray[(n,), np.dtype[bfloat16]]

        # Type used in the memory tile which aggregates across the cores
        C_memTile_ty = np.ndarray[(N_per_shimtile,), np.dtype[bfloat16]]
        tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]

        # AIE Core Function declarations

        eltwise_mul_bf16_scalar = external_func(
            "eltwise_mul_bf16_scalar", inputs=[tile_ty, tile_ty, tile_ty]
        )
        eltwise_mul_bf16_vector = external_func(
            "eltwise_mul_bf16_vector", inputs=[tile_ty, tile_ty, tile_ty]
        )

        # Tile declarations
        shims = [tile(i, 0) for i in range(n_cores)]
        MemTile = tile(0, 1)
        cores = [tile(0, 2 + i) for i in range(n_cores)]

        inA_fifos = []
        inB_fifos = []
        outC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        for i in range(n_cores):
            inA_fifos.append(
                object_fifo(f"inA{i}", shims[i], cores[i], buffer_depth, A_ty)
            )

        # Input B
        for i in range(n_cores):
            inB_fifos.append(
                object_fifo(f"inB{i}", shims[i], cores[i], buffer_depth, B_ty)
            )

        # Output C
        # Join output from the cores in a mem tile
        for i in range(n_cores):
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, buffer_depth, C_ty)
            )
        outC = object_fifo("outC", MemTile, shims[0], buffer_depth, tensor_ty)
        if n_cores > 1:
            of_offsets = [
                (np.prod(np_ndarray_type_get_shape(C_memTile_ty))) * i
                for i in range(n_cores)
            ]
        else:
            of_offsets = []
        object_fifo_link(outC_fifos, outC, of_offsets, [])

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], "mul.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        eltwise_mul_bf16_vector(elem_in_a, elem_in_b, elem_out)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            in_tasks = []
            out_tasks = []

            # Distributing host buffers (A, B) to the shim tiles
            for i in range(n_cores):
                in_tasks.append(
                    shim_dma_single_bd_task(
                        inA_fifos[i],
                        A,
                        offset=N_per_shimtile * i,
                        sizes=[1, 1, 1, N_per_shimtile],
                    )
                )
                in_tasks.append(
                    shim_dma_single_bd_task(
                        inB_fifos[i],
                        B,
                        offset=N_per_shimtile * i,
                        sizes=[1, 1, 1, N_per_shimtile],
                    )
                )
            c_task = shim_dma_single_bd_task(
                outC, C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(*in_tasks, c_task)
            dma_await_task(c_task)


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = AIEDevice.npu1_4col
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))
except ValueError:
    print("Argument is not an integer")

with mlir_mod_ctx() as ctx:
    my_eltwise_mul(dev)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
