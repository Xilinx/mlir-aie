# tile_dma.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON-level explicit per-tile DMA program.

Peer of [`Worker`][iron.Worker] (which describes the compute body of a tile).
A [`TileDma`][iron.TileDma] describes the DMA engine program for the same (or
a different) tile — what each hardware DMA channel does, which buffers
it reads/writes, and how it synchronizes with the compute side via
locks.

Used together with [`Flow`][iron.Flow] / [`PacketFlow`][iron.PacketFlow]
(which describe the AXI-stream routes) and explicit [`Buffer`][iron.Buffer]
+ [`Lock`][iron.Lock] declarations, for designs where
[`ObjectFifo`][iron.ObjectFifo] would hide too much to be useful.

A [`DmaEndpoint`][iron.DmaEndpoint] names one DMA channel of a tile. It is what
a [`DmaChannel`][iron.DmaChannel] runs on when the compiler assigns the channel,
and it builds the runtime-sequence [`TileDmaTask`][iron.TileDmaTask]s that
reprogram a mem or core tile's DMA from inside the sequence body.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    LockAction,
)
from ...dialects.aie import (
    EndOp,  # pyright: ignore[reportAttributeAccessIssue]
    _as_bd_i32,
    _as_bd_i64,
    _as_i32,
    dma_bd,  # pyright: ignore[reportAttributeAccessIssue]
    dma_start,
    mem,
    memtile_dma,
    next_bd,
    shim_mem,
    use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...dialects.aie import bds as bd_blocks
from ...dialects.aiex import dma_configure_task, dma_configure_task_for
from ...helpers.npdtypes import pack_pad_value
from ..buffer import Buffer
from ..device import Tile
from ..lock import Lock
from ..resolvable import Resolvable
from ..runtime._context import active_sequence
from ..runtime.dmataskhandle import Task

_SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)


@dataclass
class Acquire:
    """An ``aie.use_lock(..., AcquireGreaterEqual|Acquire)`` op at the start of a BD."""

    lock: Lock
    value: int = 1
    greater_equal: bool = True  # False → exact Acquire

    def emit(self) -> None:
        action = (
            LockAction.AcquireGreaterEqual if self.greater_equal else LockAction.Acquire
        )
        use_lock(self.lock.op, action, value=self.value)


@dataclass
class Release:
    """An ``aie.use_lock(..., Release)`` op at the end of a BD."""

    lock: Lock
    value: int = 1

    def emit(self) -> None:
        use_lock(self.lock.op, LockAction.Release, value=self.value)


@dataclass
class BdIteration:
    """Iteration state of a buffer descriptor for aie.dma_bd.

    Lets one BD cover ``size`` sub-buffers over ``size`` executions instead of an
    N-deep chain. Values are true/element: the base advances by ``stride``
    elements each execution and wraps after ``size`` executions; ``current`` is
    the starting step (default 0). The lowering applies the hardware ``-1`` bias
    and element->word scaling. NOTE: the identically-named ``iteration_*`` family
    on the runtime-sequence path uses RAW register values instead -- do not copy
    numbers between them.
    """

    size: int
    stride: int
    current: int = 0


@dataclass
class Bd:
    """A single buffer-descriptor entry in a [`DmaChannel`][iron.DmaChannel]'s chain.

    Lowers to one basic block containing acquires + `aie.dma_bd` +
    releases + an `aie.next_bd`. The `next` field selects what the
    `next_bd` points at:

    - `None` (default) — follow the channel: the next entry in `bds`, and from
      the last entry either back to the head or out of the chain, per
      [`DmaChannel.loop`][iron.DmaChannel].
    - `"self"` — the BD loops to itself, whatever the rest of the chain does.
    - an `int` `i` — point at the i-th BD in this channel's `bds`
      list (zero-based). Useful for explicit cycles in a multi-BD chain.

    `next` is ignored on an out-of-order channel because those BDs are chained
    only for configuration and the hardware selects by header id (see
    [`DmaChannel.out_of_order`][iron.DmaChannel]).
    """

    buffer: Buffer
    offset: int = 0
    length: int | None = None  # default: full buffer
    acquires: list[Acquire] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)
    next: int | str | None = None
    # When set, stamps a packet header on every transfer this BD emits:
    # (pkt_type, pkt_id).  Pairs with a PacketFlow that uses
    # the same pkt_id so the routing fabric dispatches correctly.
    packet: tuple[int, int] | None = None
    # Explicit id (other ids are auto-assigned around it).
    bd_id: int | None = None
    # Strided access pattern, outermost dimension first; each entry is a
    # constant int or a runtime Value. Empty (default) emits a contiguous
    # transfer. sizes and strides must have equal length.
    sizes: list = field(default_factory=list)
    strides: list = field(default_factory=list)
    # Per-BD constant-pad geometry (MemTile only): one (const_pad_before,
    # const_pad_after) pair per dimension, outermost first, matching the
    # sizes/strides layout. The fill value is per-channel (DmaChannel.pad_value).
    pad_dimensions: list[Sequence[int]] | None = None
    # BD iteration state: one BD covers N sub-buffers over N executions instead
    # of an N-deep chain. See BdIteration. Absent = iteration disabled.
    iteration: BdIteration | None = None
    # The out-of-order id stamped into the packet header. Names the slot a
    # receiving out-of-order S2MM channel places this BD's data into.
    out_of_order_id: int | None = None

    def _with_runtime_fields(self) -> Bd:
        """Return this Bd with its runtime values cast to the BD's field widths.

        A runtime descriptor's block lowers only constants, so the casts, and
        the length a runtime access pattern needs, are emitted here, ahead of
        the task.
        """
        sizes: list[Any] = [_as_bd_i64(v) for v in self.sizes]
        strides = [_as_bd_i64(v) for v in self.strides]
        length = self.length
        runtime = any(_is_value(v) for v in (*self.sizes, *self.strides, self.offset))
        if length is None and runtime:
            # The outermost of four sizes is the iteration count, which repeats
            # the transfer rather than lengthening it (as in shim_dma_bd).
            if sizes:
                length = np.prod(sizes[-3:])
            else:
                length = int(np.prod(self.buffer.shape))
        return replace(
            self,
            sizes=sizes,
            strides=strides,
            offset=_as_bd_i32(self.offset),
            length=_as_bd_i32(length),
        )

    def _emit(self, bd_id: int | None) -> None:
        """Emit the acquires, ``aie.dma_bd`` and releases at the insertion point.

        The caller supplies the block, the ``next_bd``/``aie.end`` that closes
        it, and the ``bd_id`` to stamp, which on an out-of-order channel is not
        ``self.bd_id``.
        """
        for acq in self.acquires:
            acq.emit()
        bd_kwargs: dict[str, Any] = dict(sizes=self.sizes, strides=self.strides)
        if _is_value(self.offset) or self.offset:
            bd_kwargs["offset"] = self.offset
        if self.length is not None:
            bd_kwargs["transfer_len"] = self.length
        if self.pad_dimensions is not None:
            bd_kwargs["pad_dimensions"] = self.pad_dimensions
        if self.iteration is not None:
            it = self.iteration
            bd_kwargs["iteration"] = (it.size, it.stride, it.current)
        if bd_id is not None:
            bd_kwargs["bd_id"] = bd_id
        if self.out_of_order_id is not None:
            bd_kwargs["out_of_order_id"] = self.out_of_order_id
        if self.packet is not None:
            bd_kwargs["packet"] = self.packet
        dma_bd(self.buffer.op, **bd_kwargs)
        for rel in self.releases:
            rel.emit()


class DmaEndpoint:
    """One DMA channel of a tile: its tile, direction and channel.

    ``DmaEndpoint(tile, direction, channel)`` pins a channel by index. A
    [`Flow`][iron.Flow]'s [`endpoint`][iron.dataflow.flow.Flow.endpoint] is one
    too, whose channel the compiler may assign instead.

    Pass one as a [`DmaChannel`][iron.DmaChannel]'s ``channel``, or build a
    runtime-sequence task on it with
    [`task`][iron.dataflow.tile_dma.DmaEndpoint.task].
    """

    def __init__(self, tile: Tile, direction: DMAChannelDir, channel: int | None):
        self._tile = tile
        self._direction = direction
        self._channel = channel

    @property
    def tile(self) -> Tile:
        """The tile whose DMA owns this channel."""
        return self._tile

    @property
    def direction(self) -> DMAChannelDir:
        """``S2MM`` into the tile's memory, ``MM2S`` out of it."""
        return self._direction

    @property
    def channel(self) -> int | None:
        """The channel index, or None if the compiler assigns it."""
        return self._channel

    @property
    def symbol(self) -> str | None:
        """The ``aie.route_endpoint`` a compiler-assigned channel lowers to.

        None when the channel is given by index.
        """
        return None

    def _operand(self) -> int | str:
        """Return the channel operand: its index if given, else the endpoint symbol."""
        if self.channel is not None:
            return self.channel
        assert self.symbol is not None
        return self.symbol

    def _check_on(self, tile: Tile, direction: DMAChannelDir) -> None:
        if self.tile != tile:
            raise ValueError(
                f"DMA endpoint {self} is on {self.tile}, not {tile}; a DMA "
                "program can only run its own tile's channels."
            )
        if self.direction != direction:
            raise ValueError(
                f"DMA endpoint {self} is {self.direction}, not {direction}."
            )

    def task(
        self,
        *bds: "Bd | Buffer",
        runs=1,
        wait: bool = False,
        out_of_order: bool = False,
    ) -> "TileDmaTask":
        """Configure a runtime-sequence task that walks ``bds`` on this channel.

        Call from within a [`Runtime`][iron.Runtime] sequence body on a mem or
        core tile's channel (a shim channel moves host memory: use
        ``fill``/``drain``). The buffer descriptors are written here, at the
        call; [`start`][iron.runtime.dmataskhandle.Task.start] pushes the task
        onto the channel queue, as often as needed, and
        [`free`][iron.runtime.dmataskhandle.Task.free] returns its descriptors
        after the last start.

        ```python
        load = into.endpoint(mem).task(Bd(resident, sizes=[n], strides=[1]))
        load.start()
        ...
        load.start()
        load.free()
        ```

        Each [`Bd`][iron.Bd] is spelled as in a [`TileDma`][iron.TileDma], but
        without ``next``, and its access pattern, offset and length may be
        dispatch-time values. A runtime descriptor needs a length, so one left
        unset defaults to the product of ``sizes`` (or the whole buffer). A
        buffer that has no tile yet is placed on this one.

        Args:
            *bds: The descriptors, walked in order as one task; a bare
                [`Buffer`][iron.Buffer] stands for ``Bd(buffer)``.
            runs (int | Value): How many times the whole chain runs per start.
                Defaults to 1.
            wait: Issue a completion token, so the task can be awaited.
            out_of_order: Run an S2MM channel in out-of-order mode, as
                [`DmaChannel.out_of_order`][iron.DmaChannel] does: a Bd with
                no ``bd_id`` takes its position in ``bds``, and ``runs`` counts
                packets. Needs a channel given by index.

        Returns:
            The configured [`TileDmaTask`][iron.TileDmaTask].
        """
        active = active_sequence()
        if self.tile.effective_tile_type in _SHIM_TILE_TYPES:
            raise ValueError(
                f"{self} is a shim channel, which moves host memory; use the "
                "Flow's fill()/drain() for it."
            )
        if not bds:
            raise ValueError(f"A task on {self} needs at least one Bd.")
        bds = tuple(bd if isinstance(bd, Bd) else Bd(bd) for bd in bds)
        for i, bd in enumerate(bds):
            if bd.next is not None:
                raise ValueError(
                    f"Bd {i} of a task on {self} sets next={bd.next!r}; a task "
                    "runs its Bds in order and ends after the last."
                )
            if bd.buffer.place(self.tile) != self.tile:
                raise ValueError(
                    f"A task on {self} was given a buffer on {bd.buffer.tile}; a "
                    "tile's DMA can only address buffers on that tile."
                )
            active.resolve_in_device(bd.buffer)
            for use in (*bd.acquires, *bd.releases):
                active.resolve_in_device(use.lock)
        if out_of_order and self.channel is None:
            raise ValueError(
                f"An out-of-order task needs a channel given by index, not the "
                f"compiler-assigned {self}."
            )

        bds = tuple(bd._with_runtime_fields() for bd in bds)
        if isinstance(runs, (int, np.integer)):
            repeat = dict(repeat_count=int(runs) - 1)
        else:
            repeat = dict(repeat_count_val=_as_bd_i32(runs) - _as_i32(1))
        operand = self._operand()
        if isinstance(operand, str):
            op = dma_configure_task_for(operand, issue_token=wait, **repeat)
        else:
            op = dma_configure_task(
                self.tile.op,
                self.direction,
                operand,
                issue_token=wait,
                out_of_order=out_of_order or None,
                **repeat,
            )
        with bd_blocks(op) as block:
            for i, bd in enumerate(bds):
                with block[i]:
                    bd._emit(i if out_of_order and bd.bd_id is None else bd.bd_id)
                    if i + 1 < len(bds):
                        next_bd(block[i + 1])
                    else:
                        EndOp()
        return TileDmaTask(self, op.result)

    def __str__(self) -> str:
        if self.channel is not None:
            return f"{self.direction} channel {self.channel} on {self.tile}"
        return f"{self.direction} channel on {self.tile}"


class TileDmaTask(Task):
    """A runtime-sequence DMA task on one channel of a mem or core tile.

    Built by [`DmaEndpoint.task`][iron.dataflow.tile_dma.DmaEndpoint.task],
    already configured. It is a [`Task`][iron.runtime.dmataskhandle.Task]:
    ``start()`` pushes it (again), ``await_()`` waits for it (needs
    ``wait=True``), ``free()`` returns its descriptors, and it rides a
    ``range_`` ``iter_args`` entry across loop iterations.
    """

    def __init__(self, endpoint: DmaEndpoint, handle):
        super().__init__(handle)
        self._endpoint = endpoint

    @property
    def endpoint(self) -> DmaEndpoint:
        """The channel this task runs on."""
        return self._endpoint

    def _carry(self, task: Task) -> None:
        super()._carry(task)
        if isinstance(task, TileDmaTask):
            self._endpoint = task._endpoint


@dataclass
class DmaChannel:
    """One hardware DMA channel on a tile, with its BD chain.

    Args:
        direction: `DMAChannelDir.S2MM` (host→tile) or `DMAChannelDir.MM2S`
            (tile→host).
        channel: hardware channel index, or a
            [`DmaEndpoint`][iron.DmaEndpoint] such as
            [`Flow.endpoint`][iron.dataflow.flow.Flow.endpoint]'s, to run on the
            channel the compiler assigns that end. An endpoint must be on the
            program's tile and point in ``direction``.
        bds: ordered list of [`Bd`][iron.Bd] entries that form the chain
            (in-order) or n-way merge (out-of-order).
        repeat_count: extra repeats of the task (0 = run once), where the task
            is the BD chain (in-order) or a merge round (out-of-order). Only
            meaningful on a chain that ends -- see `loop`.
        loop: whether the last BD chains back to the first (the default),
            making the chain endless. An endless chain is one task that never
            completes: it runs for as long as its locks let it, which is how
            [`ObjectFifo`][iron.ObjectFifo] expresses the same thing, and
            `repeat_count` has nothing to count and is ignored. `loop=False`
            ends the chain after its last BD, making it a task that completes
            and can be re-run -- which is what gives `repeat_count` meaning,
            and what a design reproducing a specific descriptor layout wants.
            Note a chain that ends runs exactly `repeat_count + 1` times, so a
            `loop=False` channel expected to move more than one buffer needs a
            matching count; left at 0 it moves one and stops.
        out_of_order: put the channel into out-of-order mode (S2MM only).
            Each BD receives the packet with `bd.bd_id == pkt.out_of_order_id`,
            and the BD chain (next bd) is ignored. Each BD receives its own
            `BdIteration.size` packets per merge round. Every BD must be
            packet-enabled and the ingress flow must set `keep_pkt_header=True`.
            Multiple out-of-order channels must have disjoint BD ids.
            At the hardware level, repeat_count is the total number of packets
            to accept (0-based). This class converts repeated merge rounds to
            that total.
    """

    direction: DMAChannelDir
    channel: int | DmaEndpoint
    bds: list[Bd]
    pad_value: int = 0
    repeat_count: int = 0
    out_of_order: bool = False
    # Appended rather than grouped with the chain fields above: inserting a
    # field ahead of the existing optional ones would silently rebind any
    # positional caller's argument.
    loop: bool = True

    @property
    def _key(self):
        """The channel's direction and index, or its end before assignment."""
        channel = self.channel
        if isinstance(channel, DmaEndpoint) and channel.channel is not None:
            channel = channel.channel
        return self.direction, channel

    def _pad_word(self) -> int | None:
        """Resolve the per-element pad_value into the raw 32-bit stream word.

        Returns None for the default 0 (elides the attribute). The element width
        is taken from the padded BD(s); a nonzero pad_value requires at least one
        BD with pad_dimensions (else it would silently no-op), and all padded
        BDs on the channel must share an element size (one register serves them
        all).
        """
        if not self.pad_value:
            return None
        elem_sizes = {
            np.dtype(bd.buffer.dtype).itemsize
            for bd in self.bds
            if bd.pad_dimensions is not None
        }
        if not elem_sizes:
            raise ValueError(
                "DmaChannel.pad_value is set but no BD on the channel has "
                "pad_dimensions; a pad value needs a padded region."
            )
        if len(elem_sizes) > 1:
            raise ValueError(
                "DmaChannel.pad_value is shared by all padded BDs on the channel, "
                f"but they have differing element sizes {sorted(elem_sizes)}."
            )
        return pack_pad_value(self.pad_value, elem_sizes.pop())

    def _start_repeat_count(self) -> int:
        """Lowered ``repeat_count`` for the channel's ``dma_start``.

        Out-of-order mode's hardware repeat_count is the 0-based number of
        packets to receive, so a merge of ``repeat_count + 1`` rounds lowers to
        ``packets_per_round * (repeat_count + 1) - 1``. In-order mode passes
        ``repeat_count`` through unchanged.
        """
        if self.out_of_order:
            return (
                sum(bd.iteration.size if bd.iteration else 1 for bd in self.bds)
                * (self.repeat_count + 1)
                - 1
            )
        return self.repeat_count

    def _emit_start(self, dest, chain) -> None:
        """Emit the ``aie.dma_start`` that runs this channel's chain at ``dest``."""
        channel = self.channel
        dma_start(
            self.direction,
            channel._operand() if isinstance(channel, DmaEndpoint) else channel,
            dest=dest,
            chain=chain,
            pad_value=self._pad_word() or 0,
            repeat_count=self._start_repeat_count(),
            out_of_order=self.out_of_order,
        )


def _is_value(v) -> bool:
    return v is not None and not isinstance(v, (int, np.integer))


class TileDma(Resolvable):
    """Per-tile DMA program.

    Lowers to an `aie.mem` (compute tile), `aie.memtile_dma` (memtile), or
    `aie.shim_dma` (shim tile) region based on the tile's type.

    Args:
        tile: the tile whose DMA hardware this program targets.
        channels: ordered list of [`DmaChannel`][iron.DmaChannel] entries.
    """

    def __init__(self, tile: Tile, channels: Iterable[DmaChannel]):
        self._tile = tile
        self._channels: list[DmaChannel] = []
        self.add_channels(channels)
        self._resolved = False

    @property
    def tile(self):
        return self._tile

    @property
    def channels(self) -> list[DmaChannel]:
        return list(self._channels)

    def add_channel(self, channel: DmaChannel) -> None:
        """Add a channel to this tile's DMA program.

        A tile has one DMA program, so a helper that wires transfers one at a
        time needs somewhere to put the second channel it wants on a tile it has
        already reached.
        """
        self.add_channels([channel])

    def add_channels(self, channels: Iterable[DmaChannel]) -> None:
        """Add channels after checking all hardware channel keys."""
        channels = list(channels)
        keys = {channel._key for channel in self._channels}
        for channel in channels:
            if isinstance(channel.channel, DmaEndpoint):
                channel.channel._check_on(self._tile, channel.direction)
            key = channel._key
            if key in keys:
                raise ValueError(
                    f"TileDma for {self._tile} already has "
                    f"{channel.direction} channel {channel.channel}."
                )
            keys.add(key)
        self._channels.extend(channels)

    def all_tiles(self):
        """Return this DMA's tile plus the tiles of the Buffers and Locks it uses.

        A mem tile DMA may use a neighbor's lock or buffer, whose tile must
        be resolved before that lock or buffer is.
        """
        tiles = [self._tile]
        bufs, locks = self.all_buffers_and_locks()
        for b in bufs:
            if b.tile is not None and b.tile not in tiles:
                tiles.append(b.tile)
        for lk in locks:
            if lk.tile not in tiles:
                tiles.append(lk.tile)
        return tiles

    def all_buffers_and_locks(self):
        """Iterate every Buffer + Lock this program touches.

        Program uses this to make sure they're all resolved before us.
        """
        seen_buffers: list[Buffer] = []
        seen_locks: list[Lock] = []
        for ch in self._channels:
            for bd in ch.bds:
                if bd.buffer not in seen_buffers:
                    seen_buffers.append(bd.buffer)
                for use in (*bd.acquires, *bd.releases):
                    if use.lock not in seen_locks:
                        seen_locks.append(use.lock)
        return seen_buffers, seen_locks

    def _region_decorator(self):
        """Pick the right ``aie`` region-opening decorator for the tile type.

        Asks the tile what kind it effectively is rather than reading its
        ``tile_type`` hint, which may be unset: taking that at face value
        quietly emits an ``aie.mem`` for a shim tile.
        """
        tt = self._tile.effective_tile_type
        if tt == AIETileType.MemTile:
            return memtile_dma(self._tile.op)
        if tt in (AIETileType.ShimNOCTile, AIETileType.ShimPLTile):
            return shim_mem(self._tile.op)
        # Default: compute / core tile.
        return mem(self._tile.op)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._resolved:
            return
        self._resolved = True

        decorator = self._region_decorator()

        # Layout: 2N+1 basic blocks for N channels
        #   block[0]   — entry (chain of dma_start)
        #   block[2i+1] — channel i's BD chain head
        #   block[2i+2] — channel i+1's dma_start  (or final EndOp for i+1 == N)
        # Each BD inside a channel gets its own basic block: the first BD of
        # channel i occupies block[2i+1]; subsequent BDs share the same
        # decorator-managed block sequence after all channels (handled by
        # using extra block indices for multi-BD chains).
        channels = self._channels

        def _ooo_slot_id(bd: Bd, pos: int) -> int:
            return bd.bd_id if bd.bd_id is not None else pos

        pinned_bd_ids: dict[int, int | DmaEndpoint] = {}  # slot id -> channel
        for ch in channels:
            if ch.out_of_order and ch.direction != DMAChannelDir.S2MM:
                raise ValueError(
                    "out_of_order is only valid for an S2MM DmaChannel; "
                    f"channel {ch.channel} is {ch.direction}"
                )
            if ch.out_of_order:
                if not ch.bds:
                    raise ValueError(
                        f"out_of_order channel {ch.channel} needs at least one "
                        "receive BD"
                    )
                for slot, bd in enumerate(ch.bds):
                    if bd.packet is None:
                        raise ValueError(
                            f"out_of_order channel {ch.channel} BD at slot "
                            f"{slot} must be packet-enabled"
                        )
                    pinned = _ooo_slot_id(bd, slot)
                    if pinned in pinned_bd_ids:
                        raise ValueError(
                            f"out_of_order bd_id {pinned} is used by more than "
                            f"one BD on this tile (channels "
                            f"{pinned_bd_ids[pinned]} and {ch.channel})"
                        )
                    pinned_bd_ids[pinned] = ch.channel
        if not channels:
            # Degenerate: nothing to do.  Emit an empty mem region.
            @decorator
            def _body(block):
                with block[0]:
                    EndOp()

            return

        # For multi-BD chains we need 1 + len(ch.bds) blocks per channel
        # (1 head + len(bds) actually overlapping; the first BD goes in
        # the head block, subsequent BDs in trailing blocks).  Compute
        # absolute block indices up front.
        chan_head_idx: list[int] = []  # block holding first BD per channel
        chan_extra_idx: list[list[int]] = []  # extra BD blocks per channel
        chan_chain_idx: list[int] = []  # block where next channel's dma_start sits
        next_idx = 1
        for ch in channels:
            chan_head_idx.append(next_idx)
            next_idx += 1
            extras = []
            for _ in ch.bds[1:]:
                extras.append(next_idx)
                next_idx += 1
            chan_extra_idx.append(extras)
            chan_chain_idx.append(next_idx)
            next_idx += 1
        end_idx = chan_chain_idx[-1]

        @decorator
        def _body(block):
            # Wire up dma_start chain in the entry / chain blocks.
            # Entry block: dma_start for channel 0
            channels[0]._emit_start(block[chan_head_idx[0]], block[chan_chain_idx[0]])
            # Chain blocks: dma_start for channels 1..N-1
            for i in range(1, len(channels)):
                with block[chan_chain_idx[i - 1]]:
                    channels[i]._emit_start(
                        block[chan_head_idx[i]], block[chan_chain_idx[i]]
                    )

            # Per-channel BD bodies.
            for i, ch in enumerate(channels):
                bd_block_idx = [chan_head_idx[i], *chan_extra_idx[i]]
                for bd_pos, bd in enumerate(ch.bds):
                    with block[bd_block_idx[bd_pos]]:
                        bd._emit(
                            _ooo_slot_id(bd, bd_pos) if ch.out_of_order else bd.bd_id
                        )
                        # next_bd target
                        if ch.out_of_order:
                            # Chain BDs only for configuration; the hardware
                            # ignores the chain.
                            nxt = (bd_pos + 1) % len(ch.bds)
                            next_bd(block[bd_block_idx[nxt]])
                        elif bd.next is None:
                            if bd_pos + 1 < len(ch.bds):
                                next_bd(block[bd_block_idx[bd_pos + 1]])
                            elif ch.loop:
                                next_bd(block[bd_block_idx[0]])
                            else:
                                # The region's aie.end block. A next_bd landing
                                # on it is how the dialect spells "chain ends
                                # here" -- aie-assign-bd-ids reads that as no
                                # next BD, so the task completes and
                                # repeat_count can re-run it.
                                next_bd(block[end_idx])
                        elif bd.next == "self":
                            next_bd(block[bd_block_idx[bd_pos]])
                        elif isinstance(bd.next, int):
                            if not 0 <= bd.next < len(ch.bds):
                                raise ValueError(
                                    f"Bd.next index {bd.next} out of range "
                                    f"for channel with {len(ch.bds)} BDs"
                                )
                            next_bd(block[bd_block_idx[bd.next]])
                        else:
                            raise ValueError(
                                f"Bd.next must be 'self', an int index, or None; got {bd.next!r}"
                            )

            # Final EndOp in the trailing chain block.
            with block[end_idx]:
                EndOp()
