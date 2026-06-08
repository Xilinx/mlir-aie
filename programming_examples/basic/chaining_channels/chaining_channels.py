# chaining_channels/chaining_channels.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Chaining channels — manual BD writes + per-tile DMA programs in IRON.

The teaching point is the runtime sequence's manual
``npu_writebd`` / ``npu_address_patch`` / ``npu_push_queue`` /
``npu_sync`` -- which IRON's host-side ``rt.fill`` / ``rt.drain``
abstractions hide.  Those calls live in an ``rt.inline_ops`` block.

The structural side -- per-tile DMA programs with explicit BD chains,
explicit locks, explicit ``Flow`` routes -- is expressed via IRON's
first-class lower-level primitives (``Buffer`` / ``Lock`` / ``Flow`` /
``TileDma`` / ``DmaChannel`` / ``Bd`` / ``Acquire`` / ``Release``).
These lower into the same ``aie.memtile_dma`` / ``aie.mem`` / ``aie.flow``
ops the dialect-direct version emitted by hand, but at the IRON level
where users can read + extend without dropping to the dialect.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    Compile,
    DmaChannel,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils.hostruntime.argparse import device_from_args
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aiex import (
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_write32,
    npu_writebd,
)
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace.events import (
    MemTileEvent,
    ShimTileEvent,
    MemTilePortEvent,
    ShimTilePortEvent,
)


@iron.jit
def chaining_channels(
    a_in: In,
    b_out: Out,
    *,
    length_bytes: Compile[int] = 1024,
    col: Compile[int] = 0,
    trace_size: Compile[int] = 0,
):
    # ---- types + tiles -------------------------------------------------
    n_elements = length_bytes // np.dtype(np.int32).itemsize
    n_elements_read = n_elements * 4  # 4× larger compute buffer

    vector_ty = np.ndarray[(n_elements,), np.dtype[np.int32]]
    vector_ty_read = np.ndarray[(n_elements_read,), np.dtype[np.int32]]

    shim_tile = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=col, row=1, tile_type=AIETileType.MemTile)
    compute_tile = Tile(col=col, row=2, tile_type=AIETileType.CoreTile)

    # ---- structural side: explicit Buffers + Locks + Flow + TileDma ----
    memtile_buff = Buffer(
        tile=mem_tile,
        type=vector_ty,
        name="memtile_buff",
        initial_value=np.arange(1, n_elements + 1, dtype=np.int32),
    )
    memtile_lock = Lock(tile=mem_tile, lock_id=0, init=0, name="memtile_lock")

    compute_buff = Buffer(tile=compute_tile, type=vector_ty_read, name="compute_buff")
    compute_prod_lock = Lock(
        tile=compute_tile, lock_id=0, init=1, name="compute_prod_lock"
    )
    compute_cons_lock = Lock(
        tile=compute_tile, lock_id=1, init=0, name="compute_cons_lock"
    )

    # Memtile → shim S2MM (write path); shim MM2S → compute S2MM (read path).
    mem_to_shim_flow = Flow(
        src=mem_tile,
        dst=shim_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
    )
    shim_to_compute_flow = Flow(
        src=shim_tile,
        dst=compute_tile,
        src_port=WireBundle.DMA,
        src_channel=0,
        dst_port=WireBundle.DMA,
        dst_channel=0,
    )

    # Memtile DMA: a self-chained MM2S BD that fires every time the runtime
    # sequence releases memtile_lock (lock at 0xC0000 on NPU2).
    memtile_dma = TileDma(
        tile=mem_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=memtile_buff,
                        offset=0,
                        length=n_elements,
                        acquires=[Acquire(memtile_lock, value=1)],
                        releases=[Release(memtile_lock, value=0)],
                        next="self",
                    ),
                ],
            ),
        ],
    )

    # Compute-tile DMA: a self-chained S2MM BD that fills compute_buff,
    # gated by the producer/consumer lock pair the core then flips.
    compute_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.S2MM,
                channel=0,
                bds=[
                    Bd(
                        buffer=compute_buff,
                        offset=0,
                        length=n_elements_read,
                        acquires=[Acquire(compute_prod_lock, value=1)],
                        releases=[Release(compute_cons_lock, value=1)],
                        next="self",
                    ),
                ],
            ),
        ],
    )

    # ---- compute tile spinner: toggle locks, ignore the data ----------
    def core_body(cons_lock, prod_lock):
        for _ in range_(sys.maxsize):
            cons_lock.acquire(1)
            prod_lock.release(1)

    worker = Worker(
        core_body,
        [compute_cons_lock, compute_prod_lock],
        tile=compute_tile,
        while_true=False,
    )

    # ---- runtime sequence: manual BD writes (THE lesson) ---------------
    def manual_bd_writes(a, b):
        # Release the MemTile lock to trigger the memtile MM2S BD.
        npu_write32(column=col, row=1, address=0xC0000, value=1)

        # BD 0: S2MM channel 0 on the shim (MemTile -> DDR, buffer `a`).
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

        # BD 1: MM2S channel 0 on the shim (DDR -> ComputeTile, buffer `b`).
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

        # Kick the queues.  S2MM ch0 doesn't issue a token (we don't wait on it).
        npu_push_queue(
            column=col,
            row=0,
            direction=0,
            channel=0,
            bd_id=0,
            issue_token=False,
            repeat_count=0,
        )
        # MM2S ch0 issues a token so we can sync on completion below.
        npu_push_queue(
            column=col,
            row=0,
            direction=1,
            channel=0,
            bd_id=1,
            issue_token=True,
            repeat_count=0,
        )
        npu_sync(column=col, row=0, direction=1, channel=0, column_num=1, row_num=1)

    # ---- assemble + return the Program --------------------------------
    rt = Runtime()
    rt.add_flow(mem_to_shim_flow)
    rt.add_flow(shim_to_compute_flow)
    for lk in (memtile_lock, compute_prod_lock, compute_cons_lock):
        rt.add_lock(lk)
    rt.add_tile_dma(memtile_dma)
    rt.add_tile_dma(compute_dma)

    with rt.sequence(vector_ty, vector_ty_read) as (a, b):
        if trace_size > 0:
            rt.enable_trace(
                trace_size,
                workers=[worker],
                memtile_events=[
                    MemTileEvent.LOCK_SEL0_ACQ_GE,
                    MemTilePortEvent(
                        MemTileEvent.PORT_RUNNING_0, WireBundle.South, 3, True
                    ),
                    MemTilePortEvent(
                        MemTileEvent.PORT_TLAST_0, WireBundle.South, 3, True
                    ),
                    MemTileEvent.DMA_MM2S_SEL0_STREAM_BACKPRESSURE,
                    MemTileEvent.DMA_MM2S_SEL0_STALLED_LOCK,
                    MemTileEvent.DMA_MM2S_SEL0_START_TASK,
                    MemTileEvent.DMA_MM2S_SEL0_FINISHED_TASK,
                    MemTileEvent.DMA_MM2S_SEL0_FINISHED_BD,
                ],
                shimtile_events=[
                    ShimTileEvent.DMA_S2MM_0_START_TASK,
                    ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                    ShimTileEvent.DMA_MM2S_0_START_TASK,
                    ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                    ShimTileEvent.DMA_MM2S_0_STALLED_LOCK,
                    ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
                    ShimTilePortEvent(
                        ShimTileEvent.PORT_RUNNING_0, WireBundle.South, 2, True
                    ),
                    ShimTilePortEvent(
                        ShimTileEvent.PORT_RUNNING_1, WireBundle.South, 3, False
                    ),
                ],
            )
        rt.start(worker)
        rt.inline_ops(manual_bd_writes, [a, b])

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    return dict(
        length_bytes=opts.length,
        col=opts.col,
        trace_size=opts.trace if opts.trace else 0,
    )


def main():
    p = argparse.ArgumentParser(prog="AIE Chaining Channels")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument("-n", "--length", type=int, default=1024, help="bytes (>=4)")
    p.add_argument("-c", "--col", type=int, default=0)
    p.add_argument(
        "-t", "--trace", type=int, default=0, help="trace size in bytes; 0 disables"
    )
    opts = p.parse_args()
    if opts.length % 4 != 0 or opts.length < 4:
        sys.exit(f"--length ({opts.length}) must be a positive multiple of 4")
    run_design_cli(
        chaining_channels,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
