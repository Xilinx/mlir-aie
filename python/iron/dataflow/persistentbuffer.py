# persistentbuffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""PersistentBuffer: static weight store in MemTile, streamed to compute tile via DMA.

Replaces the placed-API pattern of:
    buffer(memtile, large_type, name, initial_value=arr)
    + 4x lock()
    + flow(memtile, DMA, mm2s_ch, compute, DMA, s2mm_ch)
    + @memtile_dma (MM2S, loops forever)
    + @mem (S2MM, loops forever)

The compute tile holds a small receive buffer (full_size // repeat_count bytes).
The kernel calls acquire/release on the PersistentBufferHandle to synchronize
with the MemTile DMA — same API as an ObjectFifoHandle.
"""

import numpy as np

from ..resolvable import Resolvable


class PersistentBufferHandle:
    """Consumer handle passed as Worker fn_arg.

    acquire() acquires the comp_cons_lock (waits for DMA transfer complete).
    release() releases the comp_prod_lock (signals DMA to re-send).
    """

    def __init__(self, pb: "PersistentBuffer"):
        self._pb = pb

    def acquire(self, n: int = 1):
        """Wait for MemTile DMA to deliver the next weight chunk."""
        from ...dialects.aie import use_lock, LockAction
        use_lock(self._pb._comp_cons_lock, LockAction.AcquireGreaterEqual)
        return self._pb._recv_buf

    def release(self, n: int = 1):
        """Signal MemTile DMA to send the next weight chunk."""
        from ...dialects.aie import use_lock, LockAction
        use_lock(self._pb._comp_prod_lock, LockAction.Release)


class PersistentBuffer(Resolvable):
    """Static weight store: full data in MemTile SRAM, streamed to compute tile via DMA.

    The MemTile holds the full weight array (may be > 32KB).
    The compute tile holds a small receive buffer (full_size // repeat_count bytes).
    A DMA flow streams one chunk at a time with lock-based synchronization.

    Usage::

        pb = PersistentBuffer(
            obj_type=np.ndarray[(76800,), np.dtype[np.int8]],
            recv_type=np.ndarray[(9600,), np.dtype[np.int8]],
            initial_value=weight_array,
            name="post_l1_wts",
            repeat_count=7,
            memtile_placement=Tile(4,1),
            compute_placement=Tile(6,4),
            mm2s_channel=0,
            s2mm_channel=0,
        )
        handle = pb.cons()
        worker = Worker(fn, [handle, ...])

        # Inside fn:
        chunk = handle.acquire()   # waits for DMA, returns recv_buf
        k(chunk, ...)
        handle.release()           # signals DMA to re-send
    """

    def __init__(
        self,
        obj_type,
        initial_value,
        name: str,
        recv_type=None,
        repeat_count: int = 1,
        memtile_placement=None,
        compute_placement=None,
        mm2s_channel: int = 0,
        s2mm_channel: int = 0,
        ping_pong_buf=None,           # (obj_type, initial_value, name) for second buffer
        ping_pong_memtile=None,       # MemTile holding the second buffer (may differ)
        mem_lock_id: int = 0,         # starting lock_id for MemTile (uses id and id+1)
        comp_lock_id: int = 0,        # starting lock_id for compute tile
        pp_lock_id: int = 0,          # starting lock_id for ping-pong MemTile
    ):
        self._obj_type = obj_type
        self._initial_value = np.asarray(initial_value, dtype=np.int8)
        self._name = name
        self._recv_type = recv_type if recv_type is not None else obj_type
        self._repeat_count = repeat_count
        self._memtile = memtile_placement
        self._compute = compute_placement
        self._mm2s_ch = mm2s_channel
        self._s2mm_ch = s2mm_channel
        self._ping_pong_buf = ping_pong_buf
        self._ping_pong_memtile = ping_pong_memtile
        self._mem_lock_id = mem_lock_id
        self._comp_lock_id = comp_lock_id
        self._pp_lock_id = pp_lock_id

        # Set by resolve()
        self._comp_cons_lock = None
        self._comp_prod_lock = None
        self._recv_buf = None

    def cons(self) -> "PersistentBufferHandle":
        """Returns a consumer handle to pass as a Worker fn_arg."""
        return PersistentBufferHandle(self)

    def resolve(self, loc=None, ip=None) -> None:
        """Emit the MemTile buffer/lock/flow/DMA + compute tile DMA ops."""
        from ...dialects.aie import (
            buffer, lock, flow, memtile_dma, mem,
            dma_start, dma_bd, next_bd, use_lock,
            DMAChannelDir, LockAction, WireBundle,
            EndOp,
        )

        memtile_op = self._memtile.op
        compute_op = self._compute.op

        # --- MemTile side ---
        # Explicit lock_id required: AIEObjectFifoStatefulTransform runs before
        # AIEAssignLockIDs and requires IDs to be explicitly set.
        # For ping-pong: each buffer gets init=1 (one copy ready to send).
        # For single-buffer: init=repeat_count (pre-load N repeats).
        cons_init = 1 if self._ping_pong_buf is not None else self._repeat_count
        mem_prod_lock = lock(memtile_op, lock_id=self._mem_lock_id,     init=0)
        mem_cons_lock = lock(memtile_op, lock_id=self._mem_lock_id + 1, init=cons_init)

        wts_buf = buffer(
            memtile_op,
            self._obj_type,
            self._name,
            initial_value=self._initial_value,
        )

        # --- Compute tile side ---
        comp_prod_lock = lock(compute_op, lock_id=self._comp_lock_id,     init=1)
        comp_cons_lock = lock(compute_op, lock_id=self._comp_lock_id + 1, init=0)

        recv_buf = buffer(
            compute_op,
            self._recv_type,
            f"{self._name}_recv",
        )

        # Store for handle use
        self._comp_cons_lock = comp_cons_lock
        self._comp_prod_lock = comp_prod_lock
        self._recv_buf = recv_buf

        # --- DMA flow: MemTile MM2S → compute S2MM ---
        flow(
            memtile_op, WireBundle.DMA, self._mm2s_ch,
            compute_op,  WireBundle.DMA, self._s2mm_ch,
        )

        # --- MemTile DMA region (MM2S) ---
        if self._ping_pong_buf is not None:
            # Two-BD ping-pong: alternates between wts_buf and ping_pong second buffer.
            # The second buffer may be on an adjacent MemTile (shared memory access).
            pp_type, pp_data, pp_name = self._ping_pong_buf
            pp_memtile_op = (self._ping_pong_memtile.op
                             if self._ping_pong_memtile else memtile_op)
            pp_prod_lock = lock(pp_memtile_op, lock_id=self._pp_lock_id,     init=0)
            pp_cons_lock = lock(pp_memtile_op, lock_id=self._pp_lock_id + 1, init=1)
            pp_buf = buffer(pp_memtile_op, pp_type, pp_name,
                            initial_value=np.asarray(pp_data, dtype=np.int8))

            @memtile_dma(memtile_op)
            def _mtdma(block):
                dma_start(DMAChannelDir.MM2S, self._mm2s_ch, dest=block[1], chain=block[3])
                with block[1]:   # BD1: first buffer (FC2)
                    use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(wts_buf)
                    use_lock(mem_prod_lock, LockAction.Release)
                    next_bd(block[2])   # → BD2
                with block[2]:   # BD2: second buffer (FC1, ping-pong)
                    use_lock(pp_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(pp_buf)
                    use_lock(pp_prod_lock, LockAction.Release)
                    next_bd(block[1])   # → BD1 (alternate forever)
                with block[3]:
                    EndOp()
        else:
            @memtile_dma(memtile_op)
            def _mtdma(block):
                dma_start(DMAChannelDir.MM2S, self._mm2s_ch, dest=block[1], chain=block[2])
                with block[1]:
                    use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(wts_buf)
                    use_lock(mem_prod_lock, LockAction.Release)
                    next_bd(block[1])   # infinite loop
                with block[2]:
                    EndOp()

        # --- Compute tile DMA region (S2MM, loops forever) ---
        @mem(compute_op)
        def _cdma(block):
            dma_start(DMAChannelDir.S2MM, self._s2mm_ch, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(comp_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(recv_buf)
                use_lock(comp_cons_lock, LockAction.Release)
                next_bd(block[1])   # infinite loop
            with block[2]:
                EndOp()
