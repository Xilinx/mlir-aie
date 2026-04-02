# 07_memtile_pool_alloc/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Prototype 7: MemTile Pool Allocator
#
# Demonstrates a pool allocation pattern in MemTile: a SINGLE large buffer
# covers all sub-regions, and dma_bd(pool, offset=N, len=M) selects which
# portion of the pool each BD accesses. This proves that one contiguous
# memory allocation can serve multiple logical buffers via offset-based
# addressing in the BD configuration.
#
# Key difference from Prototype 6:
#   Proto 6: 4 separate aie.buffer() objects (data_a, data_b, result_a, result_b)
#   Proto 7: 1 pool buffer with dma_bd(pool, offset=N) for each sub-region
#
# Pool layout (2048 bytes = 2048 uint8 elements):
#   [0:256]     = data region A (input for Core A)
#   [256:512]   = data region B (input for Core B)
#   [512:768]   = result region A (output from Core A)
#   [768:1024]  = result region B (output from Core B)
#   [1024:2048] = free (available for future allocation)
#
# NOTE: Only even-numbered MemTile DMA channels are used (0, 2, 4) to
# work around a compiler BD ID allocation bug where odd channels get
# assigned BD IDs from the even-channel range (0-23 instead of 24-47).
#
# Architecture (same topology as Prototype 6):
#   DDR ──[Shim MM2S:0]──► MemTile S2MM:0 ──► pool[0:256], pool[256:512]
#                                                    │
#           MemTile MM2S:0 ◄── pool[0:256] ──────────┘  (to Core A)
#           MemTile MM2S:2 ◄── pool[256:512] ────────┘  (to Core B)
#                │                   │
#            Core(0,2)           Core(0,3)
#                │                   │
#   MemTile S2MM:2 ◄── pool[512:768]  MemTile S2MM:4 ◄── pool[768:1024]
#                                         │
#           MemTile MM2S:4 ◄──────────────┘  (BD chain: pool[512:768] then pool[768:1024])
#                │
#   DDR ◄──[Shim S2MM:0]

import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def memtile_pool_alloc(dev, chunk_size=256):
    total_size = 2 * chunk_size  # 512 bytes: 256 per core
    pool_size = 2048  # Total pool size (includes free space)
    dtype = np.dtype[np.uint8]

    # Pool region offsets (in elements = bytes for uint8)
    DATA_A_OFF = 0
    DATA_B_OFF = chunk_size
    RESULT_A_OFF = 2 * chunk_size
    RESULT_B_OFF = 3 * chunk_size

    @device(dev)
    def device_body():
        chunk_ty = np.ndarray[(chunk_size,), dtype]
        full_ty = np.ndarray[(total_size,), dtype]
        pool_ty = np.ndarray[(pool_size,), dtype]

        passthrough_fn = external_func(
            "passThroughLine",
            [chunk_ty, chunk_ty, np.int32],
            link_with="passThrough.cc.o",
        )

        # ---- Tiles ----
        ShimTile = tile(0, 0)
        MemTileTile = tile(0, 1)
        CoreA = tile(0, 2)
        CoreB = tile(0, 3)

        # ---- ONE pool buffer in MemTile (replaces 4 separate buffers) ----
        pool = buffer(MemTileTile, pool_ty, name="pool")

        # ---- MemTile locks (same as Prototype 6: 4 lock pairs) ----
        # Each pair guards one sub-region of the pool
        data_a_prod = lock(MemTileTile, lock_id=0, init=1, sym_name="data_a_prod")
        data_a_cons = lock(MemTileTile, lock_id=1, init=0, sym_name="data_a_cons")
        data_b_prod = lock(MemTileTile, lock_id=2, init=1, sym_name="data_b_prod")
        data_b_cons = lock(MemTileTile, lock_id=3, init=0, sym_name="data_b_cons")
        res_a_prod = lock(MemTileTile, lock_id=4, init=1, sym_name="res_a_prod")
        res_a_cons = lock(MemTileTile, lock_id=5, init=0, sym_name="res_a_cons")
        res_b_prod = lock(MemTileTile, lock_id=6, init=1, sym_name="res_b_prod")
        res_b_cons = lock(MemTileTile, lock_id=7, init=0, sym_name="res_b_cons")

        # ---- Core tile buffers and locks (unchanged from Prototype 6) ----
        coreA_in = buffer(CoreA, chunk_ty, name="coreA_in")
        coreA_out = buffer(CoreA, chunk_ty, name="coreA_out")
        cA_prod_in = lock(CoreA, lock_id=0, init=1, sym_name="cA_prod_in")
        cA_cons_in = lock(CoreA, lock_id=1, init=0, sym_name="cA_cons_in")
        cA_prod_out = lock(CoreA, lock_id=2, init=1, sym_name="cA_prod_out")
        cA_cons_out = lock(CoreA, lock_id=3, init=0, sym_name="cA_cons_out")

        coreB_in = buffer(CoreB, chunk_ty, name="coreB_in")
        coreB_out = buffer(CoreB, chunk_ty, name="coreB_out")
        cB_prod_in = lock(CoreB, lock_id=0, init=1, sym_name="cB_prod_in")
        cB_cons_in = lock(CoreB, lock_id=1, init=0, sym_name="cB_cons_in")
        cB_prod_out = lock(CoreB, lock_id=2, init=1, sym_name="cB_prod_out")
        cB_cons_out = lock(CoreB, lock_id=3, init=0, sym_name="cB_cons_out")

        # ---- Pre-wired flows (all even MemTile channels) ----
        # DDR → MemTile S2MM:0
        flow(ShimTile, WireBundle.DMA, 0, MemTileTile, WireBundle.DMA, 0)
        # MemTile MM2S:0 → Core A
        flow(MemTileTile, WireBundle.DMA, 0, CoreA, WireBundle.DMA, 0)
        # MemTile MM2S:2 → Core B
        flow(MemTileTile, WireBundle.DMA, 2, CoreB, WireBundle.DMA, 0)
        # Core A → MemTile S2MM:2
        flow(CoreA, WireBundle.DMA, 0, MemTileTile, WireBundle.DMA, 2)
        # Core B → MemTile S2MM:4
        flow(CoreB, WireBundle.DMA, 0, MemTileTile, WireBundle.DMA, 4)
        # MemTile MM2S:4 → DDR
        flow(MemTileTile, WireBundle.DMA, 4, ShimTile, WireBundle.DMA, 0)

        # ---- Core A: static compute + DMA program ----
        @core(CoreA)
        def core_a_body():
            for _ in range_(sys.maxsize):
                use_lock(cA_cons_in, LockAction.AcquireGreaterEqual, value=1)
                use_lock(cA_prod_out, LockAction.AcquireGreaterEqual, value=1)
                passthrough_fn(coreA_in, coreA_out, chunk_size)
                use_lock(cA_prod_in, LockAction.Release, value=1)
                use_lock(cA_cons_out, LockAction.Release, value=1)

        @mem(CoreA)
        def core_a_dma(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(cA_prod_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(coreA_in)
                use_lock(cA_cons_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(cA_cons_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(coreA_out)
                use_lock(cA_prod_out, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        # ---- Core B: static compute + DMA program ----
        @core(CoreB)
        def core_b_body():
            for _ in range_(sys.maxsize):
                use_lock(cB_cons_in, LockAction.AcquireGreaterEqual, value=1)
                use_lock(cB_prod_out, LockAction.AcquireGreaterEqual, value=1)
                passthrough_fn(coreB_in, coreB_out, chunk_size)
                use_lock(cB_prod_in, LockAction.Release, value=1)
                use_lock(cB_cons_out, LockAction.Release, value=1)

        @mem(CoreB)
        def core_b_dma(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(cB_prod_in, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(coreB_in)
                use_lock(cB_cons_in, LockAction.Release, value=1)
                next_bd(block[1])
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(cB_cons_out, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(coreB_out)
                use_lock(cB_prod_out, LockAction.Release, value=1)
                next_bd(block[3])
            with block[4]:
                EndOp()

        # ---- NO @memtile_dma — all MemTile DMA at runtime ----

        # ---- Runtime sequence (pool-based addressing) ----
        @runtime_sequence(full_ty, full_ty)
        def sequence(inTensor, outTensor):
            # ==== PHASE 1: DDR → pool[0:512] via BD chain ====
            # Shim sends 512 bytes
            t_shim_in = dma_configure_task(ShimTile, DMAChannelDir.MM2S, 0)
            with bds(t_shim_in) as bd:
                with bd[0]:
                    shim_dma_bd(
                        inTensor,
                        offset=0,
                        sizes=[1, 1, 1, total_size],
                        strides=[0, 0, 0, 1],
                    )
                    EndOp()

            # MemTile S2MM:0 — BD chain writing to pool at different offsets
            # BD[0]: 256B → pool[0:256] (data region A)
            # BD[1]: 256B → pool[256:512] (data region B)
            t_mem_recv = dma_configure_task(MemTileTile, DMAChannelDir.S2MM, 0)
            with bds(t_mem_recv) as bd:
                with bd[0]:
                    use_lock(data_a_prod, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=DATA_A_OFF, len=chunk_size)
                    use_lock(data_a_cons, LockAction.Release, value=1)
                    next_bd(bd[1])
                with bd[1]:
                    use_lock(data_b_prod, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=DATA_B_OFF, len=chunk_size)
                    use_lock(data_b_cons, LockAction.Release, value=1)
                    EndOp()

            # ==== PHASE 2: pool → Cores (different offsets into same pool) ====
            # MM2S:0 sends pool[0:256] → Core A
            t_send_a = dma_configure_task(MemTileTile, DMAChannelDir.MM2S, 0)
            with bds(t_send_a) as bd:
                with bd[0]:
                    use_lock(data_a_cons, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=DATA_A_OFF, len=chunk_size)
                    use_lock(data_a_prod, LockAction.Release, value=1)
                    EndOp()

            # MM2S:2 sends pool[256:512] → Core B
            t_send_b = dma_configure_task(MemTileTile, DMAChannelDir.MM2S, 2)
            with bds(t_send_b) as bd:
                with bd[0]:
                    use_lock(data_b_cons, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=DATA_B_OFF, len=chunk_size)
                    use_lock(data_b_prod, LockAction.Release, value=1)
                    EndOp()

            # ==== PHASE 3: Cores → pool results ====
            # S2MM:2 receives Core A result → pool[512:768]
            t_recv_a = dma_configure_task(MemTileTile, DMAChannelDir.S2MM, 2)
            with bds(t_recv_a) as bd:
                with bd[0]:
                    use_lock(res_a_prod, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=RESULT_A_OFF, len=chunk_size)
                    use_lock(res_a_cons, LockAction.Release, value=1)
                    EndOp()

            # S2MM:4 receives Core B result → pool[768:1024]
            t_recv_b = dma_configure_task(MemTileTile, DMAChannelDir.S2MM, 4)
            with bds(t_recv_b) as bd:
                with bd[0]:
                    use_lock(res_b_prod, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=RESULT_B_OFF, len=chunk_size)
                    use_lock(res_b_cons, LockAction.Release, value=1)
                    EndOp()

            # ==== PHASE 4: pool[512:1024] → DDR (drain with BD chain) ====
            # MM2S:4 chains: pool[512:768] then pool[768:1024]
            t_drain = dma_configure_task(MemTileTile, DMAChannelDir.MM2S, 4)
            with bds(t_drain) as bd:
                with bd[0]:
                    use_lock(res_a_cons, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=RESULT_A_OFF, len=chunk_size)
                    use_lock(res_a_prod, LockAction.Release, value=1)
                    next_bd(bd[1])
                with bd[1]:
                    use_lock(res_b_cons, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(pool, offset=RESULT_B_OFF, len=chunk_size)
                    use_lock(res_b_prod, LockAction.Release, value=1)
                    EndOp()

            # Shim S2MM:0 receives 512 bytes
            t_shim_out = dma_configure_task(
                ShimTile, DMAChannelDir.S2MM, 0, issue_token=True
            )
            with bds(t_shim_out) as bd:
                with bd[0]:
                    shim_dma_bd(
                        outTensor,
                        offset=0,
                        sizes=[1, 1, 1, total_size],
                        strides=[0, 0, 0, 1],
                    )
                    EndOp()

            # Start all — locks handle sequencing
            dma_start_task(t_shim_in)
            dma_start_task(t_mem_recv)
            dma_start_task(t_send_a)
            dma_start_task(t_send_b)
            dma_start_task(t_recv_a)
            dma_start_task(t_recv_b)
            dma_start_task(t_drain)
            dma_start_task(t_shim_out)
            dma_await_task(t_shim_out)

            # Free all tasks to release BD slots
            dma_free_task(t_shim_in)
            dma_free_task(t_mem_recv)
            dma_free_task(t_send_a)
            dma_free_task(t_send_b)
            dma_free_task(t_recv_a)
            dma_free_task(t_recv_b)
            dma_free_task(t_drain)


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=False, default="npu", dest="device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = AIEDevice.npu1_1col
elif opts.device == "npu2":
    dev = AIEDevice.npu2
else:
    raise ValueError(f"[ERROR] Unknown device: {opts.device}")

with mlir_mod_ctx() as ctx:
    memtile_pool_alloc(dev)
    res = ctx.module.operation.verify()
    if res is True:
        print(ctx.module)
    else:
        print(res)
