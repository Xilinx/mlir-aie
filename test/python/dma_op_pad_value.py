# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# The region-form aie.dma builder plumbs pad_value onto the channel op (MemTile
# MM2S). Check the Python dma(pad_value=) builder emits the attribute while the
# pad geometry stays on dma_bd. Targets npu2, which has the CONSTANT_PAD_VALUE
# register.

import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    buffer,
    device,
    dma,
    dma_bd,
    end,
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

        # CHECK: aie.dma(MM2S, 0) {pad_value = 7 : i32} [
        # CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
        @memtile_dma(mem_tile)
        def m():
            @dma(DMAChannelDir.MM2S, 0, pad_value=7)
            def pad_out():
                use_lock(cons, LockAction.AcquireGreaterEqual)
                dma_bd(
                    buf,
                    transfer_len=256,
                    sizes=[2],
                    strides=[128],
                    pad_dimensions=[(1, 1)],
                )
                use_lock(prod, LockAction.Release)

            end()

    print(ctx.module)
