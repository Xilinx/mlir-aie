# custom_dma/custom_dma.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, AnyComputeTile, AnyMemTile
from aie.iron.resolvable import Resolvable
from aie.dialects.aiex import set_lock_value


class ScatterReadDMA(Resolvable):
    """Read three equal-sized sections at non-uniform offsets from a MemTile buffer.

    A custom three-BD chain pattern reads ``transfer_len`` elements from ``offset_a``,
    ``offset_b``, and ``offset_c``, cycling.

    Usage: pass an instance as a Worker ``fn_arg``.  Inside the worker
    function, call ``acquire(1)`` / ``release(1)`` to synchronize with
    each BD completion.
    """

    def __init__(
        self,
        buf_type,
        initial_value: np.ndarray,
        recv_type,
        name: str,
        transfer_len: int,
        offset_a: int,
        offset_b: int,
        offset_c: int,
        memtile_placement=None,
        compute_placement=None,
    ):
        self._buf_type = buf_type
        self._initial_value = np.asarray(initial_value, dtype=np.int32)
        self._recv_type = recv_type
        self._name = name
        self._transfer_len = transfer_len
        self._offset_a = offset_a
        self._offset_b = offset_b
        self._offset_c = offset_c
        self._memtile = memtile_placement
        self._compute = compute_placement

        # Set by resolve(); used by acquire/release at kernel time.
        self._comp_cons_lock = None
        self._comp_prod_lock = None
        self._recv_buf = None

    def tiles(self) -> list:
        ts = []
        if self._memtile is not None:
            ts.append(self._memtile)
        if self._compute is not None:
            ts.append(self._compute)
        return ts

    def acquire(self, n: int = 1):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._comp_cons_lock, LockAction.AcquireGreaterEqual)
        return self._recv_buf

    def release(self, n: int = 1):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._comp_prod_lock, LockAction.Release)

    def resolve(self, loc=None, ip=None) -> None:
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
        # cons_lock starts at 0 so the DMA is blocked until the runtime
        # sequence triggers it via set_lock_value.
        mem_prod_lock = lock(memtile_op, init=0)
        mem_cons_lock = lock(memtile_op, init=0)
        self._mem_cons_lock = mem_cons_lock

        src_buf = buffer(
            memtile_op,
            self._buf_type,
            self._name,
            initial_value=self._initial_value,
        )

        # --- Compute tile side ---
        comp_prod_lock = lock(compute_op, init=1)
        comp_cons_lock = lock(compute_op, init=0)

        recv_buf = buffer(compute_op, self._recv_type, f"{self._name}_recv")

        self._comp_cons_lock = comp_cons_lock
        self._comp_prod_lock = comp_prod_lock
        self._recv_buf = recv_buf

        # --- DMA flow: MemTile MM2S → compute S2MM ---
        flow(memtile_op, WireBundle.DMA, 0, compute_op, WireBundle.DMA, 0)

        # --- MemTile DMA: three-BD chain with non-uniform offsets ---
        @memtile_dma(memtile_op)
        def _mtdma(block):
            dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[4])
            with block[1]:  # BD1: row at offset_a
                use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(src_buf, offset=self._offset_a, len=self._transfer_len)
                use_lock(mem_prod_lock, LockAction.Release)
                next_bd(block[2])
            with block[2]:  # BD2: row at offset_b
                use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(src_buf, offset=self._offset_b, len=self._transfer_len)
                use_lock(mem_prod_lock, LockAction.Release)
                next_bd(block[3])
            with block[3]:  # BD3: row at offset_c
                use_lock(mem_cons_lock, LockAction.AcquireGreaterEqual)
                dma_bd(src_buf, offset=self._offset_c, len=self._transfer_len)
                use_lock(mem_prod_lock, LockAction.Release)
                next_bd(block[1])
            with block[4]:
                EndOp()

        # --- Compute tile DMA: S2MM, loops forever ---
        @mem(compute_op)
        def _cdma(block):
            dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(comp_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(recv_buf)
                use_lock(comp_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()


def custom_dma_design(dev):
    # Buffer layout on MemTile: 4 rows × 16 columns (64 x i32).
    # Three BDs read rows 0, 1, and 3 — gaps of 1 and 2 rows (non-uniform).
    cols = 16
    total_elems = 4 * cols
    transfer_len = cols
    offset_a = 0 * cols  # row 0
    offset_b = 1 * cols  # row 1
    offset_c = 3 * cols  # row 3 (skips row 2)

    buf_type = np.ndarray[(total_elems,), np.dtype[np.int32]]
    transfer_type = np.ndarray[(transfer_len,), np.dtype[np.int32]]

    init_data = np.zeros(total_elems, dtype=np.int32)
    for r in range(4):
        init_data[r * cols : (r + 1) * cols] = np.arange(
            (r + 1) * 100, (r + 1) * 100 + cols, dtype=np.int32
        )

    # Request one MemTile and one compute tile; the placer assigns coordinates.
    memtile = AnyMemTile.copy()
    compute = AnyComputeTile.copy()

    scatter = ScatterReadDMA(
        buf_type=buf_type,
        initial_value=init_data,
        recv_type=transfer_type,
        name="scatter_buf",
        transfer_len=transfer_len,
        offset_a=offset_a,
        offset_b=offset_b,
        offset_c=offset_c,
        memtile_placement=memtile,
        compute_placement=compute,
    )

    of_out = ObjectFifo(transfer_type, depth=1, name="out")

    def core_fn(scatter_h, of_out, n):
        # Row 0
        chunk = scatter_h.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(n):
            elem_out[i] = chunk[i]
        scatter_h.release(1)
        of_out.release(1)
        # Row 1
        chunk = scatter_h.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(n):
            elem_out[i] = chunk[i]
        scatter_h.release(1)
        of_out.release(1)
        # Row 3
        chunk = scatter_h.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(n):
            elem_out[i] = chunk[i]
        scatter_h.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [scatter, of_out.prod(), transfer_len],
        tile=compute,
        while_true=True,
    )

    out_type = np.ndarray[(transfer_len * 3,), np.dtype[np.int32]]

    def rt_start_memtile_dma(scatter_obj):
        set_lock_value(scatter_obj._mem_cons_lock, 3)

    rt = Runtime()
    with rt.sequence(out_type, out_type, out_type) as (_, b_out, _):
        rt.start(worker)
        tg = rt.task_group()
        rt.drain(of_out.cons(), b_out, wait=True, task_group=tg)
        rt.inline_ops(rt_start_memtile_dma, [scatter])
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device} is unknown")

print(custom_dma_design(dev))
