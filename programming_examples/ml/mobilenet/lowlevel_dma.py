# lowlevel_dma.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""User-side helpers that drop down to placed-dialect ops inside an IRON design.

Demonstrates that arbitrary lock/BD/flow/DMA programming is possible from the
high-level IRON API by writing a Resolvable subclass. The library's existing
``isinstance(arg, Resolvable)`` walk in ``Program.resolve()`` picks these up
when they appear in a Worker's ``fn_args`` list — no library-side knowledge of
this helper is required.
"""

import numpy as np

from aie.iron.resolvable import Resolvable


class StaticWeightStream(Resolvable):
    """Stream a pre-loaded MemTile buffer to a compute tile in chunks.

    Lower-level escape hatch for cases where IRON's ``ObjectFifo`` doesn't fit
    (here: the producer is a static memory region, not a Worker). Emits raw
    placed-dialect ops:

      * MemTile holds the full weight tensor as ``aie.buffer(initial_value=...)``
      * Compute tile holds a small recv buffer
      * One ``aie.flow`` MemTile→compute, lock pair on each side
      * ``aie.memtile_dma`` on MemTile loops the source buffer forever
      * ``aie.mem`` on compute tile loops the recv buffer forever

    Pass an instance directly as a Worker ``fn_arg`` — Program's generic
    ``Resolvable`` branch picks it up. The kernel calls ``acquire(1)`` /
    ``release(1)`` on the instance to synchronize with the DMA.

    Optional ping-pong mode: pass ``ping_pong_buf`` to alternate one MM2S
    channel between two source buffers (BD chain BD1→BD2→BD1...). Used in
    mobilenet's FC pass to share one DMA channel between FC1 and FC2 weights.
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
        ping_pong_buf=None,  # (obj_type, initial_value, name) for second buffer
        ping_pong_memtile=None,  # MemTile holding the second buffer (may differ)
        mem_lock_id: int = 0,  # starting lock_id for MemTile (uses id and id+1)
        comp_lock_id: int = 0,  # starting lock_id for compute tile
        pp_lock_id: int = 0,  # starting lock_id for ping-pong MemTile
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

        # Set by resolve(); used by acquire/release at kernel time.
        self._comp_cons_lock = None
        self._comp_prod_lock = None
        self._recv_buf = None

    def tiles(self) -> list:
        """Tiles this stream depends on (memtile + compute, plus optional ping-pong memtile).

        Program.resolve() resolves these before our resolve() runs so the
        ``.op`` access in resolve() is valid.
        """
        ts = []
        if self._memtile is not None:
            ts.append(self._memtile)
        if self._compute is not None:
            ts.append(self._compute)
        if self._ping_pong_memtile is not None:
            ts.append(self._ping_pong_memtile)
        return ts

    def acquire(self, n: int = 1):
        """Wait for the next chunk to arrive. Returns the recv buffer."""
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._comp_cons_lock, LockAction.AcquireGreaterEqual)
        return self._recv_buf

    def release(self, n: int = 1):
        """Signal the MemTile DMA to send the next chunk."""
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._comp_prod_lock, LockAction.Release)

    def resolve(self, loc=None, ip=None) -> None:
        """Emit the MemTile buffer/lock/flow/DMA + compute tile DMA ops.

        Called by Program.resolve() via the generic Resolvable branch when this
        instance appears in a Worker's fn_args list.
        """
        from aie.dialects.aie import (
            buffer,
            lock,
            flow,
            memtile_dma,
            mem,
            dma_start,
            dma_bd,
            next_bd,
            use_lock,
            DMAChannelDir,
            LockAction,
            WireBundle,
            EndOp,
        )

        memtile_op = self._memtile.op
        compute_op = self._compute.op

        # --- MemTile side ---
        # Explicit lock_id required: AIEObjectFifoStatefulTransform runs before
        # AIEAssignLockIDs and requires IDs to be explicitly set when raw locks
        # coexist with ObjectFifo'd locks on the same tile.
        # For ping-pong: each buffer gets init=1 (one copy ready to send).
        # For single-buffer: init=repeat_count (pre-load N repeats).
        cons_init = 1 if self._ping_pong_buf is not None else self._repeat_count
        mem_prod_lock = lock(memtile_op, lock_id=self._mem_lock_id, init=0)
        mem_cons_lock = lock(memtile_op, lock_id=self._mem_lock_id + 1, init=cons_init)

        wts_buf = buffer(
            memtile_op,
            self._obj_type,
            self._name,
            initial_value=self._initial_value,
        )

        # --- Compute tile side ---
        comp_prod_lock = lock(compute_op, lock_id=self._comp_lock_id, init=1)
        comp_cons_lock = lock(compute_op, lock_id=self._comp_lock_id + 1, init=0)

        recv_buf = buffer(
            compute_op,
            self._recv_type,
            f"{self._name}_recv",
        )

        # Stash for acquire/release at kernel time
        self._comp_cons_lock = comp_cons_lock
        self._comp_prod_lock = comp_prod_lock
        self._recv_buf = recv_buf

        # --- DMA flow: MemTile MM2S → compute S2MM ---
        flow(
            memtile_op,
            WireBundle.DMA,
            self._mm2s_ch,
            compute_op,
            WireBundle.DMA,
            self._s2mm_ch,
        )

        # --- MemTile DMA region (MM2S) ---
        if self._ping_pong_buf is not None:
            # Two-BD ping-pong: alternates between wts_buf and the second buffer.
            # The second buffer may be on an adjacent MemTile (shared memory access).
            pp_type, pp_data, pp_name = self._ping_pong_buf
            pp_memtile_op = (
                self._ping_pong_memtile.op if self._ping_pong_memtile else memtile_op
            )
            pp_prod_lock = lock(pp_memtile_op, lock_id=self._pp_lock_id, init=0)
            pp_cons_lock = lock(pp_memtile_op, lock_id=self._pp_lock_id + 1, init=1)
            pp_buf = buffer(
                pp_memtile_op,
                pp_type,
                pp_name,
                initial_value=np.asarray(pp_data, dtype=np.int8),
            )

            @memtile_dma(memtile_op)
            def _mtdma(block):
                dma_start(
                    DMAChannelDir.MM2S, self._mm2s_ch, dest=block[1], chain=block[3]
                )
                with block[1]:  # BD1: primary buffer
                    use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(wts_buf)
                    use_lock(mem_prod_lock, LockAction.Release)
                    next_bd(block[2])
                with block[2]:  # BD2: pp buffer (alternate)
                    use_lock(pp_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(pp_buf)
                    use_lock(pp_prod_lock, LockAction.Release)
                    next_bd(block[1])
                with block[3]:
                    EndOp()

        else:

            @memtile_dma(memtile_op)
            def _mtdma(block):
                dma_start(
                    DMAChannelDir.MM2S, self._mm2s_ch, dest=block[1], chain=block[2]
                )
                with block[1]:
                    use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(wts_buf)
                    use_lock(mem_prod_lock, LockAction.Release)
                    next_bd(block[1])  # infinite loop
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
                next_bd(block[1])  # infinite loop
            with block[2]:
                EndOp()
