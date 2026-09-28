# tiledmatask.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Runtime-sequence DMA task on a mem tile or core tile.

The runtime-sequence peer of [`TileDma`][iron.TileDma], which describes a
tile's DMA program structurally and is configured once when the device is
loaded. A [`TileDmaTask`][iron.TileDmaTask] instead builds its buffer
descriptors from inside the sequence body, so its access pattern can come from
dispatch-time values and be rebuilt on every call.

That is the piece an operand held resident in a mem tile needs: the resident
buffer stays put while the descriptor reading it is re-sized per dispatch. The
shim-side verbs (``fifo.fill``/``fifo.drain``) cannot express this -- a shim
BD is the only kind that can address host DDR, and correspondingly a tile BD
can only address a buffer on its own tile.

A task is built from the [`Flow`][iron.Flow] end it drives --
``flow.task(buffer)`` for one descriptor, ``flow.chain(bds)`` for a chain
walked as one task, e.g. one per slot of a ring buffer handed back and forth
with a compute tile through locks -- so the tile, direction and channel all
come from the route. [`tile_dma_task`][iron.tile_dma_task] and
[`tile_dma_chain`][iron.tile_dma_chain] spell the same tasks out with an
explicit tile, direction and channel, and configure and start them at once.
"""

from __future__ import annotations

from typing import Callable

from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
    LockAction,
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
from ._context import require_sequence
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


def _check_buffer_tile(tile: Tile, buffer: Buffer, where: str = "") -> None:
    if buffer.tile != tile:
        raise ValueError(
            f"TileDmaTask on {tile} was given a buffer on {buffer.tile}{where}; "
            "a tile's DMA can only address buffers on that tile (only a shim BD "
            "reaches host memory)."
        )


class TileDmaTask(Task):
    """A runtime-sequence DMA task on one channel of a mem tile or core tile.

    Build one from the [`Flow`][iron.Flow] end it drives, which fixes its tile,
    direction and channel:

    ```python
    into = Flow(shim, mem)             # compiler assigns both channels
    load = into.task(resident)         # mem-side S2MM into `resident`

    def sequence(a, c):
        in_fifo.prod().fill(a)         # the shim end of `into`
        load.start()                   # write the descriptor, then push it
        ...
        load.start()                   # push the same descriptor again
    ```

    Construction checks the task and emits nothing, so a task can be declared
    next to its Flow, outside the sequence body. The first
    [`start`][iron.TileDmaTask.start] configures it (writes its buffer
    descriptors) and every start pushes it onto the channel queue, so a
    restart costs one queue push. [`configure`][iron.TileDmaTask.configure]
    does the first half alone, e.g. to write the descriptors before a loop
    whose body starts the task. Access-pattern entries that are dispatch-time
    values must exist where the task is configured, so a task that uses them
    is built inside the sequence body.

    Once configured the task is a [`Task`][iron.runtime.dmataskhandle.Task]:
    ``await_()`` (needs ``wait=True``), ``free()``, and use as a ``range_``
    ``iter_args`` entry.
    """

    def __init__(
        self,
        tile: Tile,
        direction: DMAChannelDir,
        channel: int | FlowEndpoint,
        configure: Callable[[], object],
    ):
        """Wrap an emitter; use the factories rather than this directly.

        [`Flow.task`][iron.dataflow.flow.Flow.task],
        [`Flow.chain`][iron.dataflow.flow.Flow.chain], their
        [`FlowEndpoint`][iron.FlowEndpoint] counterparts, and
        [`of_buffer`][iron.TileDmaTask.of_buffer] /
        [`of_bds`][iron.TileDmaTask.of_bds] check the task before building
        one.
        """
        super().__init__(None)
        self._tile = tile
        self._direction = direction
        self._channel = channel
        self._configure = configure

    @property
    def tile(self) -> Tile:
        """The tile whose DMA runs this task."""
        return self._tile

    @property
    def direction(self) -> DMAChannelDir:
        """``S2MM`` into the tile's memory, ``MM2S`` out of it."""
        return self._direction

    @property
    def channel(self) -> int | FlowEndpoint:
        """The channel index, or the Flow end whose channel the compiler assigns."""
        return self._channel

    @property
    def configured(self) -> bool:
        """Whether the task's buffer descriptors have been emitted."""
        return self._handle is not None

    @property
    def handle(self):
        """The configured task's ``!index`` SSA value.

        Raises:
            RuntimeError: If the task has not been configured yet.
        """
        if self._handle is None:
            raise RuntimeError(
                f"TileDmaTask on {self._tile} is not configured yet; call "
                "start() or configure() inside the runtime sequence first."
            )
        return self._handle

    def configure(self) -> "TileDmaTask":
        """Emit the task's buffer descriptors without pushing it.

        Returns:
            This task.

        Raises:
            RuntimeError: If called outside a runtime sequence body, or twice.
        """
        require_sequence("TileDmaTask.configure()")
        if self._handle is not None:
            raise RuntimeError(
                f"TileDmaTask on {self._tile} is already configured; start() "
                "pushes it again."
            )
        self._handle = self._configure().result  # type: ignore[attr-defined]
        return self

    def start(self, repeat_count: int | None = None) -> "TileDmaTask":
        """Push the task onto its channel queue, configuring it first if needed.

        ``repeat_count`` replaces the task's configured count for this start
        only.

        Returns:
            This task.

        Raises:
            RuntimeError: If called outside a runtime sequence body, or after
                ``free()``.
        """
        if self._handle is None:
            self.configure()
        super().start(repeat_count)
        return self

    @classmethod
    def of_buffer(
        cls,
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
    ) -> "TileDmaTask":
        """Build a task of one buffer descriptor over ``buffer``.

        See [`tile_dma_task`][iron.tile_dma_task] for the arguments.
        """
        _check_buffer_tile(tile, buffer)
        if (acquire is None) != (release is None):
            raise ValueError(
                "TileDmaTask needs acquire and release together, got "
                f"acquire={acquire} and release={release}; a runtime descriptor "
                "takes one of each or neither."
            )
        if (sizes is None) != (strides is None):
            raise ValueError(
                "TileDmaTask needs sizes and strides together, got "
                f"sizes={sizes} and strides={strides}"
            )
        check_flow_endpoint(tile, direction, channel)

        def configure():
            return tile_dma_single_bd_task(
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

        return cls(tile, direction, channel, configure)

    @classmethod
    def of_bds(
        cls,
        tile: Tile,
        direction: DMAChannelDir,
        channel: int | FlowEndpoint,
        bds: list[Bd],
        repeat_count=0,
        wait: bool = False,
        out_of_order: bool = False,
    ) -> "TileDmaTask":
        """Build a task of a linear chain of buffer descriptors.

        See [`tile_dma_chain`][iron.tile_dma_chain] for the arguments.
        """
        if not bds:
            raise ValueError("TileDmaTask needs at least one Bd")
        for i, bd in enumerate(bds):
            _check_buffer_tile(tile, bd.buffer, f" (Bd {i})")
            if bd.next is not None:
                raise ValueError(
                    f"TileDmaTask Bd {i} sets next={bd.next!r}; a runtime chain "
                    "runs its Bds in list order and ends after the last."
                )
            if bd.iteration is not None:
                raise ValueError(
                    f"TileDmaTask Bd {i} sets iteration; use the chain's "
                    "repeat_count or one Bd per sub-buffer instead."
                )
            locks = (len(bd.acquires), len(bd.releases))
            if locks not in ((0, 0), (1, 1)) and not (out_of_order and locks == (0, 1)):
                raise ValueError(
                    f"TileDmaTask Bd {i} has {locks[0]} acquires and {locks[1]} "
                    "releases; a runtime Bd takes one of each or neither"
                    + (", or a lone release out of order." if out_of_order else ".")
                )

        if out_of_order:
            if direction != DMAChannelDir.S2MM:
                raise ValueError(
                    f"TileDmaTask out_of_order is only valid for S2MM, not {direction}"
                )
            if isinstance(_channel_operand(channel), str):
                raise ValueError(
                    "TileDmaTask out_of_order needs a fixed channel, not the "
                    f"compiler-assigned Flow endpoint {channel}."
                )
            for i, bd in enumerate(bds):
                if bd.bd_id is None or bd.packet is None:
                    raise ValueError(
                        f"TileDmaTask out_of_order Bd {i} must set bd_id and "
                        "packet; senders address it by its bd_id."
                    )
        check_flow_endpoint(tile, direction, channel)
        bds = list(bds)

        def configure():
            if isinstance(repeat_count, int):
                rc_kwargs = dict(repeat_count=repeat_count)
            else:
                rc_kwargs = dict(repeat_count_val=_as_bd_i32(repeat_count))
            operand = _channel_operand(channel)
            if isinstance(operand, str):
                task = dma_configure_task_for(operand, issue_token=wait, **rc_kwargs)
            else:
                task = dma_configure_task(
                    tile.op,
                    direction,
                    operand,
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
            return task

        return cls(tile, direction, channel, configure)


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
) -> TileDmaTask:
    """Configure and start a DMA task on ``tile``'s ``channel``.

    Call from within a [`Runtime`][iron.Runtime] sequence body. Entries of
    ``sizes``/``strides``/``offset``/``transfer_len`` may be dispatch-time
    values, which is what lets the descriptor be rebuilt per call.
    [`Flow.task`][iron.dataflow.flow.Flow.task] builds the same task with the
    tile, direction and channel taken from the route.

    Args:
        tile: the tile whose DMA channel runs this task. ``buffer`` must live
            on it.
        direction: ``DMAChannelDir.S2MM`` or ``DMAChannelDir.MM2S``.
        channel: hardware channel index, or the
            [`FlowEndpoint`][iron.FlowEndpoint] whose channel the task runs
            on. On a mem tile the channel's parity also decides which half of
            the BD pool ``bd_id`` may come from.
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
        The configured [`TileDmaTask`][iron.TileDmaTask].
    """
    require_sequence("tile_dma_task")
    task = TileDmaTask.of_buffer(
        tile,
        direction,
        channel,
        buffer,
        sizes=sizes,
        strides=strides,
        offset=offset,
        transfer_len=transfer_len,
        wait=wait,
        packet=packet,
        bd_id=bd_id,
        acquire=acquire,
        release=release,
    )
    task.configure()
    return task.start() if start else task


def tile_dma_chain(
    tile: Tile,
    direction: DMAChannelDir,
    channel: int | FlowEndpoint,
    bds: list[Bd],
    repeat_count=0,
    wait: bool = False,
    start: bool = True,
    out_of_order: bool = False,
) -> TileDmaTask:
    """Configure and start a chain of DMA descriptors on ``tile``'s ``channel``.

    Call from within a [`Runtime`][iron.Runtime] sequence body. The chain is one
    task: its descriptors run in list order, the last one ends it, and the whole
    chain runs ``repeat_count + 1`` times. Each entry is a
    [`Bd`][iron.Bd], as in a [`TileDma`][iron.TileDma], so its locks, packet
    header and access pattern are spelled the same way.
    [`Flow.chain`][iron.dataflow.flow.Flow.chain] builds the same task with the
    tile, direction and channel taken from the route.

    A count beyond what one queue push carries is issued as several pushes of
    the same task by the compiler, and ``start(repeat_count=...)`` pushes the
    configured chain again with a different count.

    Args:
        tile: the tile whose DMA channel runs this chain. Every ``Bd``'s buffer
            must live on it.
        direction: ``DMAChannelDir.S2MM`` or ``DMAChannelDir.MM2S``.
        channel: hardware channel index, or the
            [`FlowEndpoint`][iron.FlowEndpoint] whose channel the chain runs
            on.
        bds: the descriptors, in chain order. A ``Bd`` may not set ``next``
            (the chain is linear) or ``iteration``, and takes one acquire and
            one release or neither (under ``out_of_order``, a lone release
            too).
        repeat_count (int | Value): extra runs of the whole chain (0 = once).
        wait: issue a completion token, so the returned task can be awaited.
        start: push the task onto the channel queue. ``False`` configures it
            without submitting; ``start()`` submits it later.
        out_of_order: run an S2MM channel in out-of-order mode, as
            [`DmaChannel.out_of_order`][iron.DmaChannel] does: each packet
            lands in the ``Bd`` whose ``bd_id`` matches the out-of-order id in
            its header, so every ``Bd`` pins ``bd_id`` and sets ``packet``, and
            ``repeat_count`` counts packets (0-based) rather than chain runs.
            Needs a fixed channel, not a compiler-assigned endpoint.

    Returns:
        The configured [`TileDmaTask`][iron.TileDmaTask].
    """
    require_sequence("tile_dma_chain")
    task = TileDmaTask.of_bds(
        tile,
        direction,
        channel,
        bds,
        repeat_count=repeat_count,
        wait=wait,
        out_of_order=out_of_order,
    )
    task.configure()
    return task.start() if start else task
