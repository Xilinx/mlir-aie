# tiledmatask.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Runtime-sequence DMA task on a mem tile or core tile.

The runtime-sequence peer of [`TileDma`][iron.TileDma], which describes a
tile's DMA program structurally and is configured once when the device is
loaded. A [`tile_dma_task`][iron.tile_dma_task] instead builds a buffer
descriptor from inside the sequence body, so its access pattern can come from
dispatch-time values and be rebuilt on every call.

That is the piece an operand held resident in a mem tile needs: the resident
buffer stays put while the descriptor reading it is re-sized per dispatch. The
shim-side verbs (``fifo.fill``/``fifo.drain``) cannot express this -- a shim
BD is the only kind that can address host DDR, and correspondingly a tile BD
can only address a buffer on its own tile.
"""

from __future__ import annotations

from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
    LockAction,
)
from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    dma_start_task,
)
from ...dialects.aie import _as_bd_i32
from ...dialects.aiex import tile_dma_single_bd_task
from ..buffer import Buffer
from ..dataflow.tile_dma import Acquire, Release
from ..device import Tile
from ._context import active_sequence
from .dmataskhandle import Task


def _lock_triple(use: Acquire | Release | None):
    if use is None:
        return ()
    if isinstance(use, Acquire):
        action = (
            LockAction.AcquireGreaterEqual if use.greater_equal else LockAction.Acquire
        )
        return (use.lock.op, action, use.value)
    return (use.lock.op, LockAction.Release, use.value)


def tile_dma_task(
    tile: Tile,
    direction: DMAChannelDir,
    channel: int,
    buffer: Buffer,
    sizes=None,
    strides=None,
    offset=None,
    transfer_len=None,
    wait: bool = False,
    packet: tuple[int, int] | None = None,
    bd_id: int | None = None,
    acquire: Acquire | None = None,
    release: Release | None = None,
    start: bool = True,
) -> Task:
    """Configure and start a DMA task on ``tile``'s ``channel``.

    Call from within a [`Runtime`][iron.Runtime] sequence body. Entries of
    ``sizes``/``strides``/``offset``/``transfer_len`` may be dispatch-time
    values, which is what lets the descriptor be rebuilt per call.

    Args:
        tile: the tile whose DMA channel runs this task. ``buffer`` must live
            on it.
        direction: ``DMAChannelDir.S2MM`` or ``DMAChannelDir.MM2S``.
        channel: hardware channel index. On a mem tile the channel's parity
            also decides which half of the BD pool ``bd_id`` may come from.
        buffer: the [`Buffer`][iron.Buffer] this descriptor reads or writes.
        sizes: access-pattern sizes, outermost dimension first. The outermost
            entry becomes the queue repeat count rather than a transferred
            extent, matching ``fill``/``drain``.
        strides: access-pattern strides, paired with ``sizes``.
        transfer_len: elements transferred. Required when any of
            ``sizes``/``strides``/``offset`` is a dispatch-time value: unlike
            the compile-time path, the lowering cannot infer a length from the
            buffer's shape. A dispatch-time scalar is i64, matching
            ``sizes``/``strides``; it is narrowed here to the i32 the length
            and offset fields take.
        wait: issue a completion token, so the returned task can be awaited.
        bd_id: pin the buffer descriptor id rather than letting the compiler
            allocate one.
        acquire: an [`Acquire`][iron.Acquire] emitted before the descriptor,
            for taking the buffer from a compute tile.
        release: a [`Release`][iron.Release] emitted after it.
        start: push the task onto the channel queue. ``False`` configures it
            without submitting.

    Returns:
        A [`Task`][iron.runtime.dmataskhandle.Task] carrying ``.await_()`` and
        ``.free()``.
    """
    active_sequence()  # ensure we're inside a sequence body
    if buffer.tile != tile:
        raise ValueError(
            f"tile_dma_task on {tile} was given a buffer on {buffer.tile}; a "
            "tile's DMA can only address buffers on that tile (only a shim BD "
            "reaches host memory)."
        )
    task = tile_dma_single_bd_task(
        tile.op,
        direction,
        channel,
        buffer.op,
        offset=_as_bd_i32(offset),
        sizes=sizes,
        strides=strides,
        transfer_len=_as_bd_i32(transfer_len),
        issue_token=wait,
        packet=packet,
        bd_id=bd_id,
        acquire=_lock_triple(acquire),
        release=_lock_triple(release),
    )
    if start:
        dma_start_task(task)
    return Task(task.result)
