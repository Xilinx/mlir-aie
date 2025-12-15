# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.dialects import memref


def my_vector_add():
    N = 256
    n = 16
    N_div_n = N // n

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2_1col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # ComputeTile2 elements
        # First input vector from ShimTile
        in1_cons_prod_lock = lock(ComputeTile2, lock_id=0, init=1)
        in1_cons_cons_lock = lock(ComputeTile2, lock_id=1, init=0)
        in1_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tile_ty,
            name="in1_cons_buff_0",
        )
        # Second input vector, initialized on ComputeTile2
        in2_cons_prod_lock = lock(ComputeTile2, lock_id=2, init=0)
        in2_cons_cons_lock = lock(ComputeTile2, lock_id=3, init=1)
        in2_cons_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tensor_ty,
            name="in2_cons_buff_0",
            initial_value=np.arange(N, dtype=np.int32),
        )

        # Output to ShimTile
        out_prod_lock = lock(ComputeTile2, lock_id=4, init=1)
        out_cons_lock = lock(ComputeTile2, lock_id=5, init=0)
        out_buff_0 = buffer(
            tile=ComputeTile2,
            datatype=tile_ty,
            name="out_buff_0",
        )

        # AIE-array data movement
        flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)
        flow(ComputeTile2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)

        # ComputeTile DMA configuration
        @mem(ComputeTile2)
        def m(block):
            # channel allocation in S2MM direction, channel index 0
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            # BD chains are assigned to a channel as well, where the last BD is
            # either another channel allocation or the end BD
            with block[1]:
                # wait on lock acquire
                use_lock(in1_cons_prod_lock, LockAction.AcquireGreaterEqual)
                # receive incoming data in in1_cons_buff_0 buffer
                dma_bd(in1_cons_buff_0)
                # release lock
                use_lock(in1_cons_cons_lock, LockAction.Release)
                # BD loops forever on itself
                next_bd(block[1])
            with block[2]:
                # channel allocation in MM2S direction, channel index 0
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                # BD chains are assigned to a channel as well, where the last BD is
                # either another channel allocation or the end BD
            with block[3]:
                # wait on lock acquire
                use_lock(out_cons_lock, LockAction.AcquireGreaterEqual)
                # output data from out_buff_0 buffer
                dma_bd(out_buff_0)
                # release lock
                use_lock(out_prod_lock, LockAction.Release)
                # BD loops forever on itself
                next_bd(block[3])
            with block[4]:
                EndOp()

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                use_lock(in2_cons_cons_lock, LockAction.AcquireGreaterEqual)
                for j in range_(N_div_n):
                    use_lock(in1_cons_cons_lock, LockAction.AcquireGreaterEqual)
                    use_lock(out_prod_lock, LockAction.AcquireGreaterEqual)
                    for i in range_(n):
                        out_buff_0[i] = (
                            in2_cons_buff_0[j * N_div_n + i] + in1_cons_buff_0[i]
                        )
                    use_lock(in1_cons_prod_lock, LockAction.Release)
                    use_lock(out_cons_lock, LockAction.Release)
                use_lock(in2_cons_prod_lock, LockAction.Release)

        # Allocation information for to/from AIE-array data movement (typically generated by objectfifos)
        shim_dma_allocation("of_in1", DMAChannelDir.MM2S, 0, 0)
        shim_dma_allocation("of_out", DMAChannelDir.S2MM, 0, 0)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            in1_task = shim_dma_single_bd_task("of_in1", A, sizes=[1, 1, 1, N])
            out_task = shim_dma_single_bd_task(
                "of_out", C, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in1_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in1_task)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
