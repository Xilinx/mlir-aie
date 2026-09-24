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
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    dma_bd,  # pyright: ignore[reportAttributeAccessIssue]
    dma_bd_packet,  # pyright: ignore[reportAttributeAccessIssue]
    dma_start,
    mem,
    memtile_dma,
    next_bd,
    shim_mem,
    use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...helpers.npdtypes import pack_pad_value
from ..buffer import Buffer
from ..device import Tile
from ..lock import Lock
from ..resolvable import Resolvable
from .flow import FlowEndpoint


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


@dataclass
class DmaChannel:
    """One hardware DMA channel on a tile, with its BD chain.

    Args:
        direction: `DMAChannelDir.S2MM` (host→tile) or `DMAChannelDir.MM2S`
            (tile→host).
        channel: hardware channel index, or the
            [`FlowEndpoint`][iron.FlowEndpoint] from
            [`Flow.endpoint`][iron.dataflow.flow.Flow.endpoint] to run on whichever channel
            the compiler assigns that end.
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
    channel: int | FlowEndpoint
    bds: list[Bd]
    pad_value: int = 0
    repeat_count: int = 0
    out_of_order: bool = False
    # Appended rather than grouped with the chain fields above: inserting a
    # field ahead of the existing optional ones would silently rebind any
    # positional caller's argument.
    loop: bool = True


def _emit_bd(bd: "Bd", bd_id: int | None, packet_attr: bool = False) -> None:
    """Emit one BD's acquires, packet header, ``aie.dma_bd`` and releases.

    They go at the current insertion point. The caller supplies the block and the
    ``next_bd``/``aie.end`` that closes it, and the ``bd_id`` to stamp (which on
    an out-of-order channel is not ``bd.bd_id``). ``packet_attr`` puts the packet
    header on the ``aie.dma_bd`` itself, as a runtime-sequence BD needs.
    """
    for acq in bd.acquires:
        acq.emit()
    bd_kwargs: dict[str, Any] = dict(sizes=bd.sizes, strides=bd.strides)
    if bd.offset:
        bd_kwargs["offset"] = bd.offset
    if bd.length is not None:
        bd_kwargs["transfer_len"] = bd.length
    if bd.pad_dimensions is not None:
        bd_kwargs["pad_dimensions"] = bd.pad_dimensions
    if bd.iteration is not None:
        it = bd.iteration
        bd_kwargs["iteration"] = (it.size, it.stride, it.current)
    if bd_id is not None:
        bd_kwargs["bd_id"] = bd_id
    if bd.out_of_order_id is not None:
        bd_kwargs["out_of_order_id"] = bd.out_of_order_id
    # A packet header must be a distinct aie.dma_bd_packet op placed BEFORE the
    # aie.dma_bd: the CDO/xclbin backends (AIERT / AIETargetXAIEV2) read the
    # header only from that op, not from a `packet` attribute on the dma_bd.
    # The runtime-sequence lowering is the reverse: it reads only the attribute.
    if bd.packet is not None and packet_attr:
        bd_kwargs["packet"] = bd.packet
    elif bd.packet is not None:
        pkt_type, pkt_id = bd.packet
        dma_bd_packet(pkt_type, pkt_id)
    dma_bd(bd.buffer.op, **bd_kwargs)
    for rel in bd.releases:
        rel.emit()


def _channel_pad_word(ch: "DmaChannel") -> int | None:
    """Resolve a channel's per-element pad_value into the raw 32-bit stream word.

    Returns None for the default 0 (elides the attribute). The element width is
    taken from the channel's padded BD(s); a nonzero pad_value requires at least
    one BD with pad_dimensions (else it would silently no-op), and all padded BDs
    on the channel must share an element size (one register serves them all).
    """
    if not ch.pad_value:
        return None
    elem_sizes = {
        np.dtype(bd.buffer.dtype).itemsize
        for bd in ch.bds
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
    return pack_pad_value(ch.pad_value, elem_sizes.pop())


def check_flow_endpoint(tile: Tile, direction: DMAChannelDir, channel) -> None:
    """Reject a [`FlowEndpoint`][iron.FlowEndpoint] on the wrong tile or direction.

    This catches an endpoint used on another tile or against its route's
    direction before the compiler would.
    """
    if not isinstance(channel, FlowEndpoint):
        return
    if channel.tile != tile:
        raise ValueError(
            f"Flow endpoint {channel} is on {channel.tile}, not {tile}; a DMA "
            "program can only run its own tile's channels."
        )
    if channel.direction != direction:
        raise ValueError(
            f"Flow endpoint {channel} is {channel.direction} (its Flow decides "
            f"which way it points), not {direction}."
        )


def _channel_operand(channel: "int | FlowEndpoint") -> int | str:
    """Return what ``aie.dma_start`` names: an index, or the endpoint's symbol."""
    return channel.symbol if isinstance(channel, FlowEndpoint) else channel


def _dma_start_repeat_count(ch: "DmaChannel") -> int:
    """Lowered ``repeat_count`` for a channel's ``dma_start``.

    Out-of-order mode's hardware repeat_count is the 0-based number of packets
    to receive, so a merge of ``ch.repeat_count + 1`` rounds lowers to
    ``packets_per_round * (ch.repeat_count + 1) - 1``. In-order mode passes
    ``ch.repeat_count`` through unchanged.
    """
    if ch.out_of_order:
        return (
            sum(bd.iteration.size if bd.iteration else 1 for bd in ch.bds)
            * (ch.repeat_count + 1)
            - 1
        )
    return ch.repeat_count


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
        keys = {(channel.direction, channel.channel) for channel in self._channels}
        for channel in channels:
            check_flow_endpoint(self._tile, channel.direction, channel.channel)
            key = (channel.direction, channel.channel)
            if key in keys:
                raise ValueError(
                    f"TileDma for {self._tile} already has "
                    f"{channel.direction} channel {channel.channel}."
                )
            keys.add(key)
        self._channels.extend(channels)

    def all_tiles(self):
        return [self._tile]

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

        pinned_bd_ids: dict[int, int | FlowEndpoint] = {}  # slot id -> channel
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
            ch = channels[0]
            dma_start(
                ch.direction,
                _channel_operand(ch.channel),
                dest=block[chan_head_idx[0]],
                chain=block[chan_chain_idx[0]],
                pad_value=_channel_pad_word(ch) or 0,
                repeat_count=_dma_start_repeat_count(ch),
                out_of_order=ch.out_of_order,
            )
            # Chain blocks: dma_start for channels 1..N-1
            for i in range(1, len(channels)):
                ch_i = channels[i]
                with block[chan_chain_idx[i - 1]]:
                    dma_start(
                        ch_i.direction,
                        _channel_operand(ch_i.channel),
                        dest=block[chan_head_idx[i]],
                        chain=block[chan_chain_idx[i]],
                        pad_value=_channel_pad_word(ch_i) or 0,
                        repeat_count=_dma_start_repeat_count(ch_i),
                        out_of_order=ch_i.out_of_order,
                    )

            # Per-channel BD bodies.
            for i, ch in enumerate(channels):
                bd_block_idx = [chan_head_idx[i], *chan_extra_idx[i]]
                for bd_pos, bd in enumerate(ch.bds):
                    with block[bd_block_idx[bd_pos]]:
                        _emit_bd(
                            bd,
                            (_ooo_slot_id(bd, bd_pos) if ch.out_of_order else bd.bd_id),
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
