# tile_dma.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Iron-level explicit per-tile DMA program.

Peer of :class:`Worker` (which describes the compute body of a tile).
A :class:`TileDma` describes the DMA engine program for the same (or a
different) tile — what each hardware DMA channel does, which buffers
it reads/writes, and how it synchronizes with the compute side via
locks.

Used together with :class:`Flow` / :class:`PacketFlow` (which describe
the AXI-stream routes) and explicit :class:`Buffer` + :class:`Lock`
declarations, for designs where :class:`ObjectFifo` would hide too much
to be useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import (  # type: ignore
    AIETileType,
    DMAChannelDir,
    LockAction,
)
from ...dialects.aie import (
    EndOp,  # pyright: ignore[reportAttributeAccessIssue]
    dma_bd,  # pyright: ignore[reportAttributeAccessIssue]
    dma_start,
    mem,
    memtile_dma,
    next_bd,
    shim_mem,
    use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..buffer import Buffer
from ..lock import Lock
from ..resolvable import Resolvable


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
class Bd:
    """A single buffer-descriptor entry in a :class:`DmaChannel`'s chain.

    Lowers to one basic block containing acquires + ``aie.dma_bd`` +
    releases + an ``aie.next_bd``.  The ``next`` field selects what the
    ``next_bd`` points at:

    * ``"self"`` (default) — the BD loops to itself (the common "keep
      streaming" pattern).
    * an ``int`` ``i`` — point at the i-th BD in this channel's ``bds``
      list (zero-based).  Useful for explicit cycles in a multi-BD chain.
    * ``None`` — emit no ``next_bd`` (rarely useful; this leaves the
      basic block without a terminator).
    """

    buffer: Buffer
    offset: int = 0
    length: int | None = None  # default: full buffer
    acquires: list[Acquire] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)
    next: int | str | None = "self"
    # When set, stamps a packet header on every transfer this BD emits:
    # ``(pkt_type, pkt_id)``.  Pairs with a :class:`PacketFlow` that uses
    # the same ``pkt_id`` so the routing fabric dispatches correctly.
    packet: tuple[int, int] | None = None


@dataclass
class DmaChannel:
    """One hardware DMA channel on a tile, with its BD chain.

    Args:
        direction: ``DMAChannelDir.S2MM`` (host→tile) or ``DMAChannelDir.MM2S``
            (tile→host).
        channel: hardware channel index.
        bds: ordered list of :class:`Bd` entries that form the chain.
    """

    direction: DMAChannelDir
    channel: int
    bds: list[Bd]


class TileDma(Resolvable):
    """Per-tile DMA program — lowers to an ``aie.mem`` (compute tile),
    ``aie.memtile_dma`` (memtile), or ``aie.shim_dma`` (shim tile) region
    based on the tile's type.

    Args:
        tile: the tile whose DMA hardware this program targets.
        channels: ordered list of :class:`DmaChannel` entries.
    """

    def __init__(self, tile, channels: Iterable[DmaChannel]):
        self._tile = tile
        self._channels: list[DmaChannel] = list(channels)
        self._resolved = False

    @property
    def tile(self):
        return self._tile

    def all_tiles(self):
        return [self._tile]

    def all_buffers_and_locks(self):
        """Iterate every Buffer + Lock this program touches — Program uses
        this to make sure they're all resolved before us."""
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
        """Pick the right ``aie`` region-opening decorator for the tile type."""
        tt = self._tile.tile_type
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
                ch.channel,
                dest=block[chan_head_idx[0]],
                chain=block[chan_chain_idx[0]],
            )
            # Chain blocks: dma_start for channels 1..N-1
            for i in range(1, len(channels)):
                ch_i = channels[i]
                with block[chan_chain_idx[i - 1]]:
                    dma_start(
                        ch_i.direction,
                        ch_i.channel,
                        dest=block[chan_head_idx[i]],
                        chain=block[chan_chain_idx[i]],
                    )

            # Per-channel BD bodies.
            for i, ch in enumerate(channels):
                bd_block_idx = [chan_head_idx[i], *chan_extra_idx[i]]
                for bd_pos, bd in enumerate(ch.bds):
                    with block[bd_block_idx[bd_pos]]:
                        for acq in bd.acquires:
                            acq.emit()
                        # dma_bd: pass buffer + optional offset/length.
                        # The dialect helper's signature is dma_bd(buffer, offset=, len=).
                        bd_kwargs = {}
                        if bd.offset:
                            bd_kwargs["offset"] = bd.offset
                        if bd.length is not None:
                            bd_kwargs["len"] = bd.length
                        if bd.packet is not None:
                            bd_kwargs["packet"] = bd.packet
                        dma_bd(bd.buffer.op, **bd_kwargs)
                        for rel in bd.releases:
                            rel.emit()
                        # next_bd target
                        if bd.next is None:
                            pass  # caller's problem if the block has no terminator
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
