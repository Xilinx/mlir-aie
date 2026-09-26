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

A [`tile_dma_chain`][iron.tile_dma_chain] does the same for a chain of
descriptors walked as one task, e.g. one per slot of a ring buffer handed
back and forth with a compute tile through locks.
"""

from __future__ import annotations

from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
    LockAction,
)
from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    dma_start_task,
)
from ...dialects.aie import (
    EndOp,  # pyright: ignore[reportAttributeAccessIssue]
    _as_bd_i32,
    next_bd,
)
from ...dialects.aie import bds as bd_blocks
from ...dialects.aiex import (
    dma_configure_task,
    dma_configure_task_for,
    tile_dma_single_bd_task,
)
from ..buffer import Buffer
from ..dataflow.flow import FlowEndpoint
from ..dataflow.tile_dma import (
    Acquire,
    Bd,
    Release,
    _channel_operand,
    _emit_bd,
    check_flow_endpoint,
)
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
    channel: int | FlowEndpoint,
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
        channel: hardware channel index, or the
            [`FlowEndpoint`][iron.FlowEndpoint] whose compiler-assigned
            channel the task runs on. On a mem tile the channel's parity also
            decides which half of the BD pool ``bd_id`` may come from.
        buffer: the [`Buffer`][iron.Buffer] this descriptor reads or writes.
        sizes (Sequence[int | Value], optional): access-pattern sizes, outermost
            dimension first. The outermost entry becomes the queue repeat count
            rather than a transferred
            extent, matching ``fill``/``drain``.
        strides (Sequence[int | Value], optional): access-pattern strides,
            paired with ``sizes``.
        offset (int | Value, optional): starting element offset in ``buffer``.
            Defaults to zero.
        transfer_len (int | Value, optional): elements transferred. When any
            of ``sizes``/``strides``/``offset`` is a dispatch-time value the
            lowering cannot infer a length from the buffer's shape, so it
            defaults to the product of the last three ``sizes``. An i32 or i64
            dispatch-time scalar may feed any of these: it is widened to the
            i64 of ``sizes``/``strides`` or range-checked and narrowed to the
            i32 of the length and offset fields.
        wait: issue a completion token, so the returned task can be awaited.
        packet: optional packet header as ``(packet_type, packet_id)``.
        bd_id: pin the buffer descriptor id rather than letting the compiler
            allocate one.
        acquire: an [`Acquire`][iron.Acquire] emitted before the descriptor,
            for taking the buffer from a compute tile. A runtime descriptor
            takes one acquire and one release or neither, so give ``acquire``
            and ``release`` together.
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
    if (acquire is None) != (release is None):
        raise ValueError(
            "tile_dma_task needs acquire and release together, got "
            f"acquire={acquire} and release={release}; a runtime descriptor "
            "takes one of each or neither."
        )
    check_flow_endpoint(tile, direction, channel)
    task = tile_dma_single_bd_task(
        tile.op,
        direction,
        _channel_operand(channel),
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


def tile_dma_chain(
    tile: Tile,
    direction: DMAChannelDir,
    channel: int | FlowEndpoint,
    bds: list[Bd],
    repeat_count=0,
    wait: bool = False,
    start: bool = True,
    out_of_order: bool = False,
) -> Task:
    """Configure and start a chain of DMA descriptors on ``tile``'s ``channel``.

    Call from within a [`Runtime`][iron.Runtime] sequence body. The chain is one
    task: its descriptors run in list order, the last one ends it, and the whole
    chain runs ``repeat_count + 1`` times. Each entry is a
    [`Bd`][iron.Bd], as in a [`TileDma`][iron.TileDma], so its locks, packet
    header and access pattern are spelled the same way.

    A count beyond what one queue push carries is issued as several pushes of
    the same task by the compiler, and ``Task.start(repeat_count=...)`` pushes
    the configured chain again with a different count.

    Args:
        tile: the tile whose DMA channel runs this chain. Every ``Bd``'s buffer
            must live on it.
        direction: ``DMAChannelDir.S2MM`` or ``DMAChannelDir.MM2S``.
        channel: hardware channel index, or the
            [`FlowEndpoint`][iron.FlowEndpoint] whose compiler-assigned
            channel the chain runs on.
        bds: the descriptors, in chain order. A ``Bd`` may not set ``next``
            (the chain is linear) or ``iteration``, and takes one acquire and
            one release or neither (under ``out_of_order``, a lone release
            too).
        repeat_count (int | Value): extra runs of the whole chain (0 = once).
        wait: issue a completion token, so the returned task can be awaited.
        start: push the task onto the channel queue. ``False`` configures it
            without submitting; ``Task.start()`` submits it later.
        out_of_order: run an S2MM channel in out-of-order mode, as
            [`DmaChannel.out_of_order`][iron.DmaChannel] does: each packet
            lands in the ``Bd`` whose ``bd_id`` matches the out-of-order id in
            its header, so every ``Bd`` pins ``bd_id`` and sets ``packet``, and
            ``repeat_count`` counts packets (0-based) rather than chain runs.
            Needs an integer ``channel``.

    Returns:
        A [`Task`][iron.runtime.dmataskhandle.Task] carrying ``.start()``,
        ``.await_()`` and ``.free()``.
    """
    active_sequence()  # ensure we're inside a sequence body
    if not bds:
        raise ValueError("tile_dma_chain needs at least one Bd")
    for i, bd in enumerate(bds):
        if bd.buffer.tile != tile:
            raise ValueError(
                f"tile_dma_chain on {tile} was given a buffer on {bd.buffer.tile} "
                f"(Bd {i}); a tile's DMA can only address buffers on that tile "
                "(only a shim BD reaches host memory)."
            )
        if bd.next is not None:
            raise ValueError(
                f"tile_dma_chain Bd {i} sets next={bd.next!r}; a runtime chain "
                "runs its Bds in list order and ends after the last."
            )
        if bd.iteration is not None:
            raise ValueError(
                f"tile_dma_chain Bd {i} sets iteration; use the chain's "
                "repeat_count or one Bd per sub-buffer instead."
            )
        locks = (len(bd.acquires), len(bd.releases))
        if locks not in ((0, 0), (1, 1)) and not (out_of_order and locks == (0, 1)):
            raise ValueError(
                f"tile_dma_chain Bd {i} has {locks[0]} acquires and {locks[1]} "
                "releases; a runtime Bd takes one of each or neither"
                + (", or a lone release out of order." if out_of_order else ".")
            )

    if out_of_order:
        if direction != DMAChannelDir.S2MM:
            raise ValueError(
                f"tile_dma_chain out_of_order is only valid for S2MM, not {direction}"
            )
        if isinstance(channel, FlowEndpoint):
            raise ValueError(
                "tile_dma_chain out_of_order needs an integer channel, not the "
                f"Flow endpoint {channel}."
            )
        for i, bd in enumerate(bds):
            if bd.bd_id is None or bd.packet is None:
                raise ValueError(
                    f"tile_dma_chain out_of_order Bd {i} must set bd_id and "
                    "packet; senders address it by its bd_id."
                )

    if isinstance(repeat_count, int):
        rc_kwargs = dict(repeat_count=repeat_count)
    else:
        rc_kwargs = dict(repeat_count_val=_as_bd_i32(repeat_count))
    check_flow_endpoint(tile, direction, channel)
    if isinstance(channel, FlowEndpoint):
        task = dma_configure_task_for(channel.symbol, issue_token=wait, **rc_kwargs)
    else:
        task = dma_configure_task(
            tile.op,
            direction,
            channel,
            issue_token=wait,
            out_of_order=out_of_order,
            **rc_kwargs,
        )
    with bd_blocks(task) as block:
        for i, bd in enumerate(bds):
            with block[i]:
                _emit_bd(bd, bd.bd_id, packet_attr=True)
                if i + 1 < len(bds):
                    next_bd(block[i + 1])
                else:
                    EndOp()
    if start:
        dma_start_task(task)
    return Task(task.result)
