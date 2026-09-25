# flow.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON-level circuit- and packet-switched route primitives.

Two classes live here: [`Flow`][iron.Flow] (circuit-switched) and
[`PacketFlow`][iron.PacketFlow] (packet-switched, with explicit packet IDs),
plus the small [`PacketDest`][iron.PacketDest] dataclass PacketFlow uses for
its destination list. They share a private `_emit_shim_dma_alloc`
helper and are treated as a sibling pair by `dataflow/__init__.py`;
splitting them across two modules would either duplicate the helper
or require a third file to hold it.

Both are peers of [`ObjectFifo`][iron.ObjectFifo] in the dataflow namespace.
ObjectFifo wraps *route + buffers + locks + DMA* into one
circular-buffer abstraction; `Flow` / `PacketFlow` are the
lower-level "just declare the route" primitives, paired with explicit
[`TileDma`][iron.TileDma] programs (and [`Buffer`][iron.Buffer] /
[`Lock`][iron.Lock] shared state) for designs that need direct control.
"""

from dataclasses import dataclass
from typing import Sequence

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from ...dialects.aie import (
    flow as _flow_op,
)
from ...dialects.aie import (
    packetflow as _packetflow_op,
)
from ...dialects.aie import (
    shim_dma_allocation,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..device import Tile  # noqa: F401  (re-exported via package)
from ..resolvable import NotResolvedError, Resolvable

_SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)


def _default_shim_symbol(kind: str, src, src_channel, dst, dst_channel) -> str:
    """Name the shim channel this route ends at.

    Named after the channel, which is what lets ``fill`` / ``drain`` work
    without the caller inventing a symbol and repeating it at both ends.

    That names the channel, not the route: two routes sharing one shim channel
    (a broadcast off a single MM2S, say) derive the same name and collide at
    symbol definition. Pass an explicit ``shim_symbol`` for those.
    """
    if src.effective_tile_type in _SHIM_TILE_TYPES:
        shim, direction, channel = src, "mm2s", src_channel
    elif dst.effective_tile_type in _SHIM_TILE_TYPES:
        shim, direction, channel = dst, "s2mm", dst_channel
    else:
        raise ValueError(
            f"{kind} has no shim endpoint to transfer to or from: neither src "
            f"({src}) nor dst ({dst}) is a shim tile."
        )
    if shim.col is None or shim.row is None:
        raise ValueError(
            f"{kind} cannot name its shim channel because {shim} is not fully "
            "placed; pass an explicit shim_symbol."
        )
    return f"shim_{shim.col}_{shim.row}_{direction}_{channel}"


def _emit_shim_dma_alloc(
    kind: str, shim_symbol, src, src_channel, dst, dst_channel, *, loc=None, ip=None
):
    if src.effective_tile_type in _SHIM_TILE_TYPES:
        shim_dma_allocation(
            shim_symbol, src.op, DMAChannelDir.MM2S, src_channel, loc=loc, ip=ip
        )
    elif dst.effective_tile_type in _SHIM_TILE_TYPES:
        shim_dma_allocation(
            shim_symbol, dst.op, DMAChannelDir.S2MM, dst_channel, loc=loc, ip=ip
        )
    else:
        raise ValueError(
            f"{kind}.shim_symbol={shim_symbol!r} requires a shim endpoint, "
            f"but neither src ({src}) nor dst ({dst}) is a shim tile."
        )


class Flow(Resolvable):
    """An explicit AXI-stream route between a source and destination endpoint.

    Connects ``(src_tile, src_port, src_channel)`` to
    ``(dst_tile, dst_port, dst_channel)``.
    Lowers to a single `aie.flow` op. The user is responsible for
    arranging matching [`TileDma`][iron.TileDma] channels on the producer and
    consumer ends.
    """

    def __init__(
        self,
        src: Tile,
        dst: Tile,
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
            shim_symbol (str | None): Name for the ``aie.shim_dma_allocation``
                this Flow emits for its shim endpoint. Only needed to refer to
                the channel by name from elsewhere (e.g. a raw
                ``shim_dma_single_bd_task("symbol", ...)``); ``fill``/``drain``
                name it themselves. Direction is inferred: shim-as-source →
                MM2S, shim-as-dest → S2MM.
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
        """Return the tiles this Flow touches — Program uses this to resolve them."""
        return [self._src, self._dst]

    def _transfer(self, rt_data, direction, **kwargs):
        """Emit a transfer referencing the allocation emitted by resolve()."""
        from ..runtime._context import active_sequence
        from ..runtime.dmatask import emit_shim_transfer

        if self not in active_sequence()._runtime.flows:
            raise ValueError(
                f"{type(self).__name__} must be registered with "
                "rt.add_flow(flow) before the runtime sequence fills or drains it."
            )
        src_is_shim = self._src.effective_tile_type in _SHIM_TILE_TYPES
        dst_is_shim = self._dst.effective_tile_type in _SHIM_TILE_TYPES
        if src_is_shim and dst_is_shim:
            raise ValueError(
                "Flow.fill()/drain() require exactly one shim endpoint; "
                "shim-to-shim transfers need explicit endpoint allocations."
            )
        if direction == DMAChannelDir.MM2S and not src_is_shim:
            raise ValueError(
                "fill() sends data into the array, so it needs a Flow whose "
                f"src is a shim tile; this one's src is {self._src}. "
                "To read results back out, use drain()."
            )
        if direction == DMAChannelDir.S2MM and not dst_is_shim:
            raise ValueError(
                "drain() reads results back out of the array, so it needs a "
                f"Flow whose dst is a shim tile; this one's dst is {self._dst}. "
                "To send data in, use fill()."
            )
        if self._shim_symbol is None:
            self._shim_symbol = _default_shim_symbol(
                type(self).__name__,
                self._src,
                self._src_channel,
                self._dst,
                self._dst_channel,
            )
        return emit_shim_transfer(self._shim_symbol, rt_data, **kwargs)

    def fill(self, source, **kwargs):
        """Send data from the ``source`` runtime buffer into this route.

        Call from within a [`Runtime`][iron.Runtime] sequence body, on a Flow
        with exactly one shim endpoint, at the source. See ``emit_shim_transfer``
        for the keyword arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        return self._transfer(source, DMAChannelDir.MM2S, **kwargs)

    def drain(self, dest, **kwargs):
        """Receive data from this route into the ``dest`` runtime buffer.

        Call from within a [`Runtime`][iron.Runtime] sequence body, on a Flow
        with exactly one shim endpoint, at the destination. See
        ``emit_shim_transfer`` for the keyword arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        return self._transfer(dest, DMAChannelDir.S2MM, **kwargs)

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
                loc=loc,
                ip=ip,
            )
            if self._shim_symbol is not None:
                _emit_shim_dma_alloc(
                    "Flow",
                    self._shim_symbol,
                    self._src,
                    self._src_channel,
                    self._dst,
                    self._dst_channel,
                    loc=loc,
                    ip=ip,
                )


@dataclass
class PacketDest:
    """One destination endpoint of a [`PacketFlow`][iron.PacketFlow].

    Held as a small dataclass so the PacketFlow constructor's destination list
    reads cleanly when there are multiple sinks (uncommon, but the underlying op
    supports it).
    """

    tile: Tile
    port: WireBundle = WireBundle.DMA
    channel: int = 0


class PacketFlow(Resolvable):
    """An explicit packet-switched route from a source to one or more destinations.

    Connects ``(src_tile, src_port, src_channel)`` to each destination
    endpoint, tagging the stream with `pkt_id`. Lowers to a single
    `aie.packetflow` op holding one `aie.packet_source` and one
    `aie.packet_dest` per destination. The user is responsible for
    arranging matching [`TileDma`][iron.TileDma] channels on the producer and
    consumer ends.
    """

    def __init__(
        self,
        pkt_id: int,
        src: Tile,
        dst: Tile,
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
            src: Source tile.
            dst: Primary destination tile.
            src_port: Source port bundle (as for [`Flow`][iron.Flow]).
            src_channel: Source channel (as for [`Flow`][iron.Flow]).
            dst_port: Destination port bundle (as for [`Flow`][iron.Flow]).
            dst_channel: Destination channel (as for [`Flow`][iron.Flow]).
            extra_dsts: Additional destination endpoints if this packet needs
                to fan out. Each is a [`PacketDest`][iron.PacketDest].
            keep_pkt_header: If `True`, downstream tile receives the 4-byte
                packet header alongside the payload (useful when the receiver
                needs to re-emit with the same pkt_id). Defaults to `False`.
            shim_symbol: Same meaning as on [`Flow`][iron.Flow] — auto-emit a
                matching `aie.shim_dma_allocation` when one endpoint is a
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
            loc=loc,
            ip=ip,
        )
        if self._shim_symbol is not None:
            _emit_shim_dma_alloc(
                "PacketFlow",
                self._shim_symbol,
                self._src,
                self._src_channel,
                self._dst,
                self._dst_channel,
                loc=loc,
                ip=ip,
            )
