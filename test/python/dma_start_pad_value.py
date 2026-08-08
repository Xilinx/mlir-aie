# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The terminator-form dma_start() builder plumbs pad_value onto aie.dma_start
# (MemTile MM2S); the pad geometry stays on dma_bd. Targets npu2, which has the
# CONSTANT_PAD_VALUE register.

import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    EndOp,
    LockAction,
    buffer,
    device,
    dma_bd,
    dma_start,
    lock,
    memtile_dma,
    tile,
    use_lock,
)
from aie.extras.context import mlir_mod_ctx

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2_1col)
    def device_body():
        mem_tile = tile(0, 1)
        prod = lock(mem_tile, lock_id=0, init=1)
        cons = lock(mem_tile, lock_id=1, init=0)
        ty = np.ndarray[(256,), np.dtype[np.int32]]
        buf = buffer(tile=mem_tile, datatype=ty, name="mem")

        # CHECK: aie.dma_start(MM2S, 0, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}) {pad_value = 7 : i32}
        # CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
        @memtile_dma(mem_tile)
        def m(block):
            # S2MM fills the buffer; MM2S drains it, padding on the channel.
            dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(prod, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf, transfer_len=256)
                use_lock(cons, LockAction.Release, value=1)
                EndOp()
            with block[2]:
                dma_start(
                    DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4], pad_value=7
                )
            with block[3]:
                use_lock(cons, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(
                    buf,
                    transfer_len=256,
                    sizes=[2],
                    strides=[128],
                    pad_dimensions=[(1, 1)],
                )
                use_lock(prod, LockAction.Release, value=1)
                EndOp()
            with block[4]:
                EndOp()

    print(ctx.module)
