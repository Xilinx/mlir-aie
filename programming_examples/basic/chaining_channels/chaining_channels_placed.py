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
import aie.utils.trace as trace_utils

N = 1024  # 1kB buffer (256 int32 elements = 1024 bytes)
dev = AIEDevice.npu2_1col
col = 0  # Always use column 0
trace_size = 8192  # Trace buffer size in bytes
enable_trace = 0  # Trace disabled by default

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    enable_trace = int(sys.argv[3])


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
            memtile_lock = lock(MemTile, lock_id=0, init=0, sym_name="memtile_lock")

            # ComputeTile buffer and locks for read data
            compute_buff = buffer(
                tile=ComputeTile2,
                datatype=vector_ty_read,
                name="compute_buff"
            )
            compute_prod_lock = lock(ComputeTile2, lock_id=0, init=1, sym_name="compute_prod_lock")
            compute_cons_lock = lock(ComputeTile2, lock_id=1, init=0, sym_name="compute_cons_lock")

            # Flow from MemTile to ShimTile for write path
            flow(MemTile, WireBundle.DMA, 0, ShimTile, WireBundle.DMA, 0)
            
            # Flow from ShimTile to ComputeTile for read path
            flow(ShimTile, WireBundle.DMA, 0, ComputeTile2, WireBundle.DMA, 0)

            # MemTile DMA logic - send initialized data to DDR when triggered
            @memtile_dma(MemTile)
            def memtile_dma_block(block):
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
                s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(compute_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(compute_buff, offset=0, len=n_elements_read)
                    use_lock(compute_cons_lock, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[2]:
                    EndOp()

            # Compute tile core - toggle locks, discard data
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    use_lock(compute_cons_lock, LockAction.AcquireGreaterEqual, value=1)
                    # Do nothing with the data, just toggle locks
                    use_lock(compute_prod_lock, LockAction.Release, value=1)

            # Configure tracing on ShimTile (if enabled)
            if enable_trace:
                tiles_to_trace = [ShimTile]
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

            # Runtime sequence
            @runtime_sequence(vector_ty, vector_ty_read)
            def sequence(A, B):
                # Setup trace buffer (if enabled)
                if enable_trace:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile,
                        trace_size=trace_size,
                    )

                # Release MemTile lock to trigger DMA
                npu_write32(column=col, row=1, address=0xC0000, value=1)
                
                # Write BD for S2MM channel 0 (MemTile -> DDR, buffer A)
                npu_writebd(
                    bd_id=0,
                    buffer_length=n_elements,
                    buffer_offset=0,
                    column=col,
                    row=0,
                    enable_packet=0,
                    out_of_order_id=0,
                    packet_id=0,
                    packet_type=0,
                    d0_size=0,
                    d0_stride=0,
                    d0_zero_before=0,
                    d0_zero_after=0,
                    d1_size=0,
                    d1_stride=0,
                    d1_zero_before=0,
                    d1_zero_after=0,
                    d2_size=0,
                    d2_stride=0,
                    d2_zero_before=0,
                    d2_zero_after=0,
                    iteration_current=0,
                    iteration_size=0,
                    iteration_stride=0,
                    lock_acq_enable=1,
                    lock_acq_id=0,
                    lock_acq_val=0,
                    lock_rel_id=1,
                    lock_rel_val=1,
                    next_bd=0,
                    use_next_bd=0,
                    valid_bd=1,
                )
                npu_address_patch(addr=0x1D004, arg_idx=0, arg_plus=0)
                
                # Write BD for MM2S channel 0 (DDR -> ComputeTile, buffer B)
                npu_writebd(
                    bd_id=1,
                    buffer_length=n_elements_read,
                    buffer_offset=0,
                    column=col,
                    row=0,
                    enable_packet=0,
                    out_of_order_id=0,
                    packet_id=0,
                    packet_type=0,
                    d0_size=0,
                    d0_stride=0,
                    d0_zero_before=0,
                    d0_zero_after=0,
                    d1_size=0,
                    d1_stride=0,
                    d1_zero_before=0,
                    d1_zero_after=0,
                    d2_size=0,
                    d2_stride=0,
                    d2_zero_before=0,
                    d2_zero_after=0,
                    iteration_current=0,
                    iteration_size=0,
                    iteration_stride=0,
                    lock_acq_enable=1,
                    lock_acq_id=1,
                    lock_acq_val=1,
                    lock_rel_id=0,
                    lock_rel_val=0,
                    next_bd=0,
                    use_next_bd=0,
                    valid_bd=1,
                )
                npu_address_patch(addr=0x1D024, arg_idx=1, arg_plus=0)
                
                # Push BD 0 to S2MM channel 0 queue
                npu_push_queue(column=col, row=0, direction=0, channel=0, bd_id=0, issue_token=False, repeat_count=0)
                # Wait for S2MM channel 0
                # npu_sync(column=col, row=0, direction=0, channel=0, column_num=1, row_num=1)
                
                # Push BD 1 to MM2S channel 0 queue
                npu_push_queue(column=col, row=0, direction=1, channel=0, bd_id=1, issue_token=True, repeat_count=0)
                # Wait for MM2S channel 0
                npu_sync(column=col, row=0, direction=1, channel=0, column_num=1, row_num=1)

                if enable_trace:
                    trace_utils.gen_trace_done_aie2(ShimTile)

    print(ctx.module)


my_chaining_channels()
