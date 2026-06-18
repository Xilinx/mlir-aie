# flow.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Iron-level circuit- and packet-switched route primitives.

Two classes live here: :class:`Flow` (circuit-switched) and
:class:`PacketFlow` (packet-switched, with explicit packet IDs), plus
the small :class:`PacketDest` dataclass PacketFlow uses for its
destination list.  They share a private ``_emit_shim_dma_alloc``
helper and are treated as a sibling pair by ``dataflow/__init__.py``;
splitting them across two modules would either duplicate the helper
or require a third file to hold it.

Both are peers of :class:`ObjectFifo` in the dataflow namespace.
ObjectFifo wraps *route + buffers + locks + DMA* into one
circular-buffer abstraction; ``Flow`` / ``PacketFlow`` are the
lower-level "just declare the route" primitives, paired with explicit
:class:`TileDma` programs (and :class:`Buffer` / :class:`Lock`
shared state) for designs that need direct control.
"""

from dataclasses import dataclass
from typing import Sequence

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import (  # type: ignore
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from ...dialects.aie import (
    flow as _flow_op,
    packetflow as _packetflow_op,
    shim_dma_allocation,
)
from ..device import Tile  # noqa: F401  (re-exported via package)
from ..resolvable import NotResolvedError, Resolvable

_SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)


def _emit_shim_dma_alloc(kind: str, shim_symbol, src, src_channel, dst, dst_channel):
    if src.tile_type in _SHIM_TILE_TYPES:
        shim_dma_allocation(shim_symbol, src.op, DMAChannelDir.MM2S, src_channel)
    elif dst.tile_type in _SHIM_TILE_TYPES:
        shim_dma_allocation(shim_symbol, dst.op, DMAChannelDir.S2MM, dst_channel)
    else:
        raise ValueError(
            f"{kind}.shim_symbol={shim_symbol!r} requires a shim endpoint, "
            f"but neither src ({src}) nor dst ({dst}) is a shim tile."
        )


class Flow(Resolvable):
    """An explicit AXI-stream route between (src_tile, src_port, src_channel) and
    (dst_tile, dst_port, dst_channel).

    Lowers to a single ``aie.flow`` op.  The user is responsible for
    arranging matching :class:`TileDma` channels on the producer and
    consumer ends.
    """

    def __init__(
        self,
        src,
        dst,
        *,
        src_port: WireBundle = WireBundle.DMA,
        src_channel: int = 0,
        dst_port: WireBundle = WireBundle.DMA,
        dst_channel: int = 0,
        shim_symbol: str | None = None,
    ):
        """Construct a Flow.

        Args:
            src (Tile): The source tile.
            dst (Tile): The destination tile.
            src_port (WireBundle): The source port bundle.  Defaults to DMA.
            src_channel (int): The source channel.  Defaults to 0.
            dst_port (WireBundle): The destination port bundle.  Defaults to DMA.
            dst_channel (int): The destination channel.  Defaults to 0.
            shim_symbol (str | None): When this Flow has a shim endpoint that
                will be driven from the runtime sequence (via
                ``shim_dma_single_bd_task("symbol", ...)``), provide the
                symbol name here and the Flow will emit a matching
                ``aie.shim_dma_allocation`` at the device level.  Direction
                is inferred: shim-as-source → MM2S, shim-as-dest → S2MM.
        """
        self._src = src
        self._dst = dst
        self._src_port = src_port
        self._src_channel = src_channel
        self._dst_port = dst_port
        self._dst_channel = dst_channel
        self._shim_symbol = shim_symbol
        self._op = None

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def all_tiles(self):
        """The tiles this Flow touches — Program uses this to resolve them."""
        return [self._src, self._dst]

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is None:
            self._op = _flow_op(
                self._src.op,
                self._src_port,
                self._src_channel,
                self._dst.op,
                self._dst_port,
                self._dst_channel,
            )
            if self._shim_symbol is not None:
                _emit_shim_dma_alloc(
                    "Flow",
                    self._shim_symbol,
                    self._src,
                    self._src_channel,
                    self._dst,
                    self._dst_channel,
                )


@dataclass
class PacketDest:
    """One destination endpoint of a :class:`PacketFlow`.  Held as a small
    dataclass so the PacketFlow constructor's destination list reads cleanly
    when there are multiple sinks (uncommon, but the underlying op supports
    it).
    """

    tile: object  # Tile (avoid forward-ref import)
    port: WireBundle = WireBundle.DMA
    channel: int = 0


class PacketFlow(Resolvable):
    """An explicit packet-switched route with caller-controlled ``pkt_id``.

    Peer of :class:`Flow` for the packet-switched case.  Unlike the
    ``--packet-sw-objFifos`` global lowering (which auto-assigns sequential
    packet IDs to every ObjectFifo in the design), :class:`PacketFlow`
    exposes the packet ID directly so the same ID can be reused across
    stages and used as a routing decision (e.g. memtile dispatch by
    ``pkt_id`` to one of several compute cores).

    Lowers to a single ``aie.packetflow`` op containing one
    ``aie.packet_source`` and one or more ``aie.packet_dest`` ops in its
    region.
    """

    def __init__(
        self,
        pkt_id: int,
        src,
        dst,
        *,
        src_port: WireBundle = WireBundle.DMA,
        src_channel: int = 0,
        dst_port: WireBundle = WireBundle.DMA,
        dst_channel: int = 0,
        extra_dsts: Sequence[PacketDest] = (),
        keep_pkt_header: bool = False,
        shim_symbol: str | None = None,
    ):
        """Construct a PacketFlow.

        Args:
            pkt_id: The packet ID — the same byte the routing fabric uses to
                dispatch.  Caller controls the value (often reused across
                stages so a memtile can re-emit packets keeping the original
                ID for downstream routing).
            src, dst: Source / primary destination tiles.
            src_port / src_channel / dst_port / dst_channel: As for :class:`Flow`.
            extra_dsts: Additional destination endpoints if this packet needs
                to fan out.  Each is a :class:`PacketDest`.
            keep_pkt_header: If ``True``, downstream tile receives the 4-byte
                packet header alongside the payload (useful when the receiver
                needs to re-emit with the same pkt_id).  Defaults to ``False``.
            shim_symbol: Same meaning as on :class:`Flow` — auto-emit a
                matching ``aie.shim_dma_allocation`` when one endpoint is a
                shim tile.
        """
        self._pkt_id = pkt_id
        self._src = src
        self._dst = dst
        self._src_port = src_port
        self._src_channel = src_channel
        self._dst_port = dst_port
        self._dst_channel = dst_channel
        self._extra_dsts: list[PacketDest] = list(extra_dsts)
        self._keep_pkt_header = keep_pkt_header
        self._shim_symbol = shim_symbol
        self._op = None

    @property
    def pkt_id(self) -> int:
        return self._pkt_id

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def all_tiles(self):
        tiles = [self._src, self._dst]
        tiles.extend(d.tile for d in self._extra_dsts)
        return tiles

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is not None:
            return
        dests = [
            {"dest": self._dst.op, "port": self._dst_port, "channel": self._dst_channel}
        ]
        for d in self._extra_dsts:
            dests.append({"dest": d.tile.op, "port": d.port, "channel": d.channel})
        self._op = _packetflow_op(
            pkt_id=self._pkt_id,
            source=self._src.op,
            source_port=self._src_port,
            source_channel=self._src_channel,
            dests=dests,
            keep_pkt_header=self._keep_pkt_header,
        )
        if self._shim_symbol is not None:
            _emit_shim_dma_alloc(
                "PacketFlow",
                self._shim_symbol,
                self._src,
                self._src_channel,
                self._dst,
                self._dst_channel,
            )
