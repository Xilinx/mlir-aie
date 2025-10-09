# chaining_channels/chaining_channels_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

N = 1024  # 1kB buffer (256 int32 elements = 1024 bytes)
dev = AIEDevice.npu2_1col
col = 0  # Always use column 0
verify = 0  # Optional verification path

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    verify = int(sys.argv[3])


def my_chaining_channels():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            # Convert N from bytes to int32 elements
            n_elements = N // 4
            vector_ty = np.ndarray[(n_elements,), np.dtype[np.int32]]
            
            # Read buffer is 4x larger (4KB)
            n_elements_read = (N * 4) // 4
            vector_ty_read = np.ndarray[(n_elements_read,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile2 = tile(col, 2)

            # MemTile buffer and locks for initialized data
            memtile_buff = buffer(
                tile=MemTile,
                datatype=vector_ty,
                name="memtile_buff",
                initial_value=np.arange(1, n_elements + 1, dtype=np.int32)
            )
            # Lock ID 0 is located at address 0xC0000 on NPU2
            # Initialize to 0, will be released by npu_write32 in runtime sequence
            memtile_lock = lock(MemTile, lock_id=0, init=0, sym_name="memtile_lock")

            # ComputeTile buffer and locks for read data
            compute_buff = buffer(
                tile=ComputeTile2,
                datatype=vector_ty_read,
                name="compute_buff"
            )
            # prod_lock init=1 (buffer available for DMA to write)
            # cons_lock init=0 (no data for core to read yet)
            compute_prod_lock = lock(ComputeTile2, lock_id=0, init=1, sym_name="compute_prod_lock")
            compute_cons_lock = lock(ComputeTile2, lock_id=1, init=0, sym_name="compute_cons_lock")

            if verify:
                # Verification buffer and locks
                verify_buff = buffer(
                    tile=ComputeTile2,
                    datatype=vector_ty_read,
                    name="verify_buff"
                )
                # prod_lock init=1 (buffer available for core to write)
                # cons_lock init=0 (no data for DMA to send yet)
                verify_prod_lock = lock(ComputeTile2, lock_id=2, init=1, sym_name="verify_prod_lock")
                verify_cons_lock = lock(ComputeTile2, lock_id=3, init=0, sym_name="verify_cons_lock")

            # Flow from MemTile to ShimTile for write path
            flow(MemTile, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)
            
            # Flow from ShimTile to ComputeTile for read path
            flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)
            
            if verify:
                # Flow from ComputeTile to ShimTile for verification path
                flow(ComputeTile2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 1)

            # MemTile DMA logic - send initialized data to DDR when triggered
            @memtile_dma(MemTile)
            def memtile_dma_block(block):
                # MM2S channel to send data to ShimTile
                s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(memtile_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(memtile_buff, offset=0, len=n_elements)
                    use_lock(memtile_lock, LockAction.Release, value=0)
                    next_bd(block[1])
                with block[2]:
                    EndOp()

            # ComputeTile DMA logic
            @mem(ComputeTile2)
            def compute_dma_block(block):
                # S2MM channel to receive data from ShimTile
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(compute_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(compute_buff, offset=0, len=n_elements_read)
                    use_lock(compute_cons_lock, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    if verify:
                        # MM2S channel to send verification data to ShimTile
                        s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
                    else:
                        EndOp()
                if verify:
                    with block[3]:
                        use_lock(verify_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                        dma_bd(verify_buff, offset=0, len=n_elements_read)
                        use_lock(verify_prod_lock, LockAction.Release, value=1)
                        next_bd(block[3])
                    with block[4]:
                        EndOp()

            # Compute tile core - handle read data
            @core(ComputeTile2)
            def core_body():
                # Loop indefinitely to handle data
                for _ in range_(sys.maxsize):
                    use_lock(compute_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                    if verify:
                        # Copy data to verification buffer
                        use_lock(verify_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                        for i in range_(n_elements_read):
                            verify_buff[i] = compute_buff[i]
                        use_lock(verify_cons_lock, LockAction.Release, value=1)
                    use_lock(compute_prod_lock, LockAction.Release, value=1)

            # To/from AIE-array data movement using DMA task operations
            if verify:
                @runtime_sequence(vector_ty, vector_ty_read, vector_ty_read)
                def sequence(A, B, C):
                    # Release MemTile lock to trigger DMA
                    npu_write32(column=col, row=1, address=0xC0000, value=1)
                    
                    # Task to read from MemTile to DDR (1KB)
                    write_task = dma_configure_task(ShimTile, DMAChannelDir.S2MM, 0, issue_token=True)
                    with bds(write_task) as bd:
                        with bd[0]:
                            shim_dma_bd(A, offset=0, sizes=[1, 1, 1, n_elements], strides=[0, 0, 0, 1])
                            EndOp()
                    
                    # Task to write from DDR to ComputeTile (4KB)
                    read_task = dma_configure_task(ShimTile, DMAChannelDir.MM2S, 0, issue_token=True)
                    with bds(read_task) as bd:
                        with bd[0]:
                            shim_dma_bd(B, offset=0, sizes=[1, 1, 1, n_elements_read], strides=[0, 0, 0, 1])
                            EndOp()
                    
                    # Task to read verification data from ComputeTile to DDR (4KB)
                    verify_task = dma_configure_task(ShimTile, DMAChannelDir.S2MM, 1, issue_token=True)
                    with bds(verify_task) as bd:
                        with bd[0]:
                            shim_dma_bd(C, offset=0, sizes=[1, 1, 1, n_elements_read], strides=[0, 0, 0, 1])
                            EndOp()
                    
                    # Start write task first
                    dma_start_task(write_task)
                    dma_await_task(write_task)
                    
                    # Then start read task
                    dma_start_task(read_task)
                    dma_await_task(read_task)

                    # Then start verify task
                    dma_start_task(verify_task)
                    dma_await_task(verify_task)
            else:
                @runtime_sequence(vector_ty, vector_ty_read)
                def sequence(A, B):
                    # Release MemTile lock to trigger DMA
                    npu_write32(column=col, row=1, address=0xC0000, value=1)
                    
                    # Task to read from MemTile to DDR (1KB)
                    write_task = dma_configure_task(ShimTile, DMAChannelDir.S2MM, 0, issue_token=True)
                    with bds(write_task) as bd:
                        with bd[0]:
                            shim_dma_bd(A, offset=0, sizes=[1, 1, 1, n_elements], strides=[0, 0, 0, 1])
                            EndOp()
                    
                    # Task to write from DDR to ComputeTile (4KB)
                    read_task = dma_configure_task(ShimTile, DMAChannelDir.MM2S, 0, issue_token=True)
                    with bds(read_task) as bd:
                        with bd[0]:
                            shim_dma_bd(B, offset=0, sizes=[1, 1, 1, n_elements_read], strides=[0, 0, 0, 1])
                            EndOp()
                    
                    # Start write task first
                    dma_start_task(write_task)
                    dma_await_task(write_task)
                    
                    # Then start read task
                    dma_start_task(read_task)
                    dma_await_task(read_task)

    print(ctx.module)


my_chaining_channels()
