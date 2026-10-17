# flow.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""IRON-level circuit- and packet-switched route primitives.

Two classes live here: [`Flow`][iron.Flow] (circuit-switched) and
[`PacketFlow`][iron.PacketFlow] (packet-switched, with explicit packet IDs),
plus the small [`PacketDest`][iron.PacketDest] dataclass PacketFlow uses for
its destination list, and [`FlowEndpoint`][iron.FlowEndpoint], one end of a
Flow as the DMA on that end's tile sees it. Flow and PacketFlow share how they
name their ends and how the runtime sequence fills or drains a shim end.

Both are peers of [`ObjectFifo`][iron.ObjectFifo] in the dataflow namespace.
ObjectFifo wraps *route + buffers + locks + DMA* into one
circular-buffer abstraction; `Flow` / `PacketFlow` are the
lower-level "just declare the route" primitives, paired with explicit
[`TileDma`][iron.TileDma] programs (and [`Buffer`][iron.Buffer] /
[`Lock`][iron.Lock] shared state) for designs that need direct control.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    DMAChannelDir,
    WireBundle,
)
from ...dialects._aie_ops_gen import (  # pyright: ignore[reportMissingImports]
    FlowOp as _FlowOp,
)
from ...dialects.aie import (
    flow as _flow_op,
)
from ...dialects.aie import (
    packetflow as _packetflow_op,
)
from ...dialects.aie import (
    route as _route_op,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...dialects.aie import (
    route_endpoint as _route_endpoint_op,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...dialects.aie import (
    shim_dma_allocation,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...helpers.sourceloc import SourceSite
from ..device import Tile  # noqa: F401  (re-exported via package)
from ..resolvable import NotResolvedError, PerDeviceConfigurationResolvable
from ..runtime._context import active_sequence
from ..runtime.dmatask import emit_shim_transfer
from .tile_dma import _SHIM_TILE_TYPES, DmaEndpoint


class _Route(PerDeviceConfigurationResolvable):
    """What Flow and PacketFlow share: named ends, endpoints and shim transfers.

    Subclasses set ``_src``, ``_dsts``, ``_name``, ``_shim_symbol``,
    ``_endpoints`` and the per-end ``_channels``.
    """

    _broadcast = False
    _src: Tile
    _dsts: list[Tile]
    _name: str | None
    _op: Any
    _shim_symbol: str | None
    _shim_ends: set[int]
    _channels: list[int | None]
    _endpoints: "dict[int, FlowEndpoint]"

    def _endpoint_tiles(self):
        return [self._src, *self._dsts]

    def all_tiles(self):
        """Return the tiles this route touches — Program uses this to resolve them."""
        return self._endpoint_tiles()

    @property
    def name(self) -> str | None:
        """The name the route's end symbols are built from.

        None for a route left unnamed until its configuration resolves.
        """
        return self._name

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    @property
    def _routed(self) -> bool:
        """Whether this route lowers to route endpoints rather than one op."""
        return False

    def endpoint(self, tile: Tile) -> "FlowEndpoint":
        """Return this route's end on ``tile``.

        Pass it as a [`DmaChannel`][iron.DmaChannel]'s ``channel``, or build a
        runtime-sequence task on it with
        [`task`][iron.dataflow.tile_dma.DmaEndpoint.task]. Its channel is the
        one given to the route, or else the one the compiler assigns.
        """
        ends = [i for i, t in enumerate(self._endpoint_tiles()) if t == tile]
        if len(ends) != 1:
            raise ValueError(
                f"{tile} is {'not an end' if not ends else 'more than one end'} "
                f"of this {type(self).__name__}."
            )
        end = ends[0]
        if end not in self._endpoints:
            self._endpoints[end] = FlowEndpoint(self, end)
        return self._endpoints[end]

    def _shim_end(self) -> int | None:
        """Return the end ``shim_symbol`` names: a shim source, else a lone shim dst."""
        if self._src.effective_tile_type in _SHIM_TILE_TYPES:
            return 0
        if len(self._dsts) == 1 and self._dsts[0].effective_tile_type in (
            _SHIM_TILE_TYPES
        ):
            return 1
        return None

    def _end_symbol(self, end: int) -> str:
        if end == self._shim_end() and self._shim_symbol is not None:
            return self._shim_symbol
        if self._name is None:
            raise ValueError(
                f"This {type(self).__name__} has no name yet: an unnamed route "
                "is named by the configuration it is registered with "
                "(rt.add_flow) when that resolves."
            )
        if end == 0:
            return f"{self._name}_src"
        return f"{self._name}_dst{end - 1}" if self._broadcast else f"{self._name}_dst"

    def _allocated_ends(self) -> list[int]:
        """Return the shim ends a transfer used, and the one ``shim_symbol`` names."""
        ends = set(self._shim_ends)
        if self._shim_symbol is not None:
            end = self._shim_end()
            if end is None:
                raise ValueError(
                    f"{type(self).__name__}.shim_symbol={self._shim_symbol!r} "
                    f"requires a shim endpoint, but neither src ({self._src}) nor "
                    "its one dst is a shim tile."
                )
            ends.add(end)
        return sorted(ends)

    def _emit_shim_dma_alloc(
        self, loc: ir.Location | None, ip: ir.InsertionPoint | None
    ) -> None:
        """Name the shim channels of a route that gives its channels."""
        for end in self._allocated_ends():
            shim_dma_allocation(
                self._end_symbol(end),
                self._endpoint_tiles()[end].op,
                DMAChannelDir.MM2S if end == 0 else DMAChannelDir.S2MM,
                self._channels[end],
                loc=loc,
                ip=ip,
            )

    def _transfer(self, rt_data, direction, **kwargs):
        """Emit a fill's MM2S at the source or a drain's S2MM at the one dst."""
        kind = type(self).__name__
        if self not in active_sequence()._runtime.flows:
            raise ValueError(
                f"{kind} must be registered with rt.add_flow(flow) before the "
                "runtime sequence fills or drains it."
            )
        if direction == DMAChannelDir.MM2S:
            if self._src.effective_tile_type not in _SHIM_TILE_TYPES:
                raise ValueError(
                    f"fill() sends data into the array, so it needs a {kind} whose "
                    f"src is a shim tile; this one's src is {self._src}. "
                    "To read results back out, use drain()."
                )
            end = 0
        else:
            if (
                len(self._dsts) != 1
                or self._dsts[0].effective_tile_type not in _SHIM_TILE_TYPES
            ):
                dsts = ", ".join(str(d) for d in self._dsts)
                raise ValueError(
                    "drain() reads results back out of the array, so it needs a "
                    f"{kind} whose one dst is a shim tile; this one's dst is "
                    f"{dsts}. To send data in, use fill()."
                )
            end = 1
        self._shim_ends.add(end)
        return emit_shim_transfer(self._end_symbol(end), rt_data, **kwargs)

    def drain(self, dest, **kwargs):
        """Receive data from this route into the ``dest`` runtime buffer.

        Call from within a [`Runtime`][iron.Runtime] sequence body, on a route
        whose one destination is a shim tile; the source may be a shim too. See
        ``emit_shim_transfer`` for the keyword arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        return self._transfer(dest, DMAChannelDir.S2MM, **kwargs)


class FlowEndpoint(DmaEndpoint):
    """One end of a Flow or PacketFlow, seen from the DMA on that end's tile.

    Obtained from [`Flow.endpoint`][iron.dataflow.flow.Flow.endpoint] or
    [`PacketFlow.endpoint`][iron.dataflow.flow.PacketFlow.endpoint]. The end
    knows its tile, its direction (MM2S at the source, S2MM at a destination)
    and its channel: the one the route gives, or else the one the compiler
    assigns. Like any [`DmaEndpoint`][iron.DmaEndpoint] it stands in for a
    [`DmaChannel`][iron.DmaChannel]'s ``channel`` and builds runtime-sequence
    tasks with [`task`][iron.dataflow.tile_dma.DmaEndpoint.task].

    It is not an [`ObjectFifo`][iron.ObjectFifo] endpoint: those are the
    Workers and runtime a fifo attaches to and places, whereas this names one
    DMA channel of a route between tiles the design already has.
    """

    def __init__(self, flow: _Route, end: int):
        super().__init__(
            flow._endpoint_tiles()[end],
            DMAChannelDir.MM2S if end == 0 else DMAChannelDir.S2MM,
            flow._channels[end],
        )
        self._flow = flow
        self._end = end

    def sync(self) -> None:
        """Wait until this end's tasks have issued their completion tokens.

        Called in a [`Runtime`][iron.Runtime] sequence body, as
        ``DmaEndpoint.sync`` is.

        A shim end the sequence waits on is driven at run time, so it gets its
        ``aie.shim_dma_allocation``, as an end a ``fill`` or ``drain`` uses
        does, and no fifo is given the channel. That is the end a route's
        other runtime traffic programs, such as a gather's descriptors pushed
        by control packets.

        Raises:
            ValueError: The route is not registered with the runtime, or the
                tile is not placed or the channel is not given.
        """
        flow = self._flow
        if flow not in active_sequence()._runtime.flows:
            raise ValueError(
                f"{type(flow).__name__} must be registered with rt.add_flow(flow) "
                "before the runtime sequence waits on its end."
            )
        super().sync()
        if self.tile.effective_tile_type in _SHIM_TILE_TYPES:
            flow._shim_ends.add(self._end)

    @property
    def symbol(self) -> str | None:
        """The ``aie.route_endpoint`` this end lowers to.

        None for a Flow that gives both channels, which lowers to an
        ``aie.flow`` with no endpoint symbols.
        """
        return self._flow._end_symbol(self._end) if self._flow._routed else None

    def __str__(self) -> str:
        if self.channel is None and self._flow.name is not None:
            return f"@{self.symbol}"
        return super().__str__()


class Flow(_Route):
    """An explicit AXI-stream route from a source to one or more destinations.

    Connects ``(src_tile, src_port, src_channel)`` to
    ``(dst_tile, dst_port, dst_channel)``. With both channels given, it lowers
    to a single `aie.flow` op, and the user arranges matching
    [`TileDma`][iron.TileDma] channels on the producer and consumer ends.

    A channel left as ``None`` (the default) is assigned by the compiler: the
    Flow then lowers to one ``aie.route_endpoint`` per end and an
    ``aie.route``, and a DMA program reaches an end through
    [`endpoint`][iron.dataflow.flow.Flow.endpoint] rather than an index. That is
    also how a list of destinations (a circuit-switched broadcast) lowers.
    """

    def __init__(
        self,
        src: Tile,
        dst: Tile | Sequence[Tile],
        *,
        src_port: WireBundle = WireBundle.DMA,
        src_channel: int | None = None,
        dst_port: WireBundle = WireBundle.DMA,
        dst_channel: int | None = None,
        vias: Sequence[
            tuple[
                Tile,
                tuple[WireBundle, int],
                tuple[WireBundle, int],
            ]
        ] = (),
        shim_symbol: str | None = None,
        name: str | None = None,
    ):
        """Construct a Flow.

        Args:
            src (Tile): The source tile.
            dst (Tile | Sequence[Tile]): The destination tile, or several to
                broadcast to.
            src_port (WireBundle): The source port bundle.  Defaults to DMA.
            src_channel (int | None): The source channel. ``None`` (default)
                lets the compiler assign a DMA channel.
            dst_port (WireBundle): The destination port bundle.  Defaults to DMA.
            dst_channel (int | None): The channel at every destination. ``None``
                (default) lets the compiler assign each a DMA channel.
            vias: Exact ``(tile, ingress, egress)`` switchbox hops. Each port is
                a ``(WireBundle, channel)`` tuple. Vias require one destination
                and explicit source and destination channels.
            shim_symbol (str | None): Name the runtime sequence reaches the
                shim end by, in place of the one built from ``name``. Only
                needed to refer to the channel from elsewhere (e.g. a raw
                ``shim_dma_single_bd_task("symbol", ...)``); ``fill``/``drain``
                name it themselves. Direction is inferred: shim-as-source →
                MM2S, shim-as-dest → S2MM. When both ends are shim tiles it
                names the source; a drain at the destination uses
                ``{name}_dst``.
            name (str | None): Base of the end symbols, ``{name}_src`` and
                ``{name}_dst`` (``{name}_dst{i}`` for a broadcast). Left
                ``None``, the configuration the Flow is registered with names
                it ``flow{n}`` when it resolves, counting in registration order.
        """
        self._site = SourceSite.capture()
        self._broadcast = not isinstance(dst, Tile)
        self._dsts: list[Tile] = [dst] if isinstance(dst, Tile) else list(dst)
        if not self._dsts:
            raise ValueError("Flow needs at least one destination.")
        self._vias = list(vias)
        if self._vias and (
            self._broadcast or src_channel is None or dst_channel is None
        ):
            raise ValueError(
                "Flow vias require one destination and explicit source and "
                "destination channels."
            )
        for port, channel, end in (
            (src_port, src_channel, "src"),
            (dst_port, dst_channel, "dst"),
        ):
            if channel is None and port != WireBundle.DMA:
                raise ValueError(
                    f"Flow {end}_port={port} needs an explicit {end}_channel; "
                    "the compiler only assigns DMA channels."
                )
        self._src = src
        self._src_port = src_port
        self._dst_port = dst_port
        self._channels = [src_channel] + [dst_channel] * len(self._dsts)
        self._shim_symbol = shim_symbol
        self._shim_ends: set[int] = set()
        self._name = name
        self._endpoints: dict[int, FlowEndpoint] = {}
        self._op = None

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return list(self._dsts) if self._broadcast else self._dsts[0]

    @property
    def _routed(self) -> bool:
        """Whether this Flow lowers to route endpoints rather than `aie.flow`."""
        return self._broadcast or None in self._channels

    def all_tiles(self):
        tiles = self._endpoint_tiles()
        return [
            *tiles,
            *[tile for tile, _, _ in self._vias if tile not in tiles],
        ]

    def fill(self, source, **kwargs):
        """Send data from the ``source`` runtime buffer into this route.

        Call from within a [`Runtime`][iron.Runtime] sequence body, on a Flow
        whose source is a shim tile; a destination may be a shim too. See
        ``emit_shim_transfer`` for the keyword arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        return self._transfer(source, DMAChannelDir.MM2S, **kwargs)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is not None:
            return
        loc = loc or self._site.location(self._name)
        if self._routed:
            self._resolve_route(loc, ip)
            return
        if self._vias:
            self._op = _FlowOp(
                self._src.op,
                self._src_port,
                self._channels[0],
                self._dsts[0].op,
                self._dst_port,
                self._channels[1],
                [tile.op for tile, _, _ in self._vias],
                via_ingress_bundles=[ingress[0].value for _, ingress, _ in self._vias],
                via_ingress_channels=[ingress[1] for _, ingress, _ in self._vias],
                via_egress_bundles=[egress[0].value for _, _, egress in self._vias],
                via_egress_channels=[egress[1] for _, _, egress in self._vias],
                loc=loc,
                ip=ip,
            )
        else:
            self._op = _flow_op(
                self._src.op,
                self._src_port,
                self._channels[0],
                self._dsts[0].op,
                self._dst_port,
                self._channels[1],
                loc=loc,
                ip=ip,
            )
        if self._shim_symbol is not None or self._shim_ends:
            self._emit_shim_dma_alloc(loc, ip)

    def _resolve_route(
        self, loc: ir.Location | None, ip: ir.InsertionPoint | None
    ) -> None:
        """Emit one ``aie.route_endpoint`` per end and the ``aie.route`` joining them.

        A shim end carries ``fifoName``, which is what gives it the shim DMA
        allocation the runtime sequence's transfers are renamed to: the end
        ``shim_symbol`` would name, and each end a transfer used.
        """
        allocated = self._shim_ends | {self._shim_end()}
        ports = [self._src_port] + [self._dst_port] * len(self._dsts)
        for end, (tile, port, channel) in enumerate(
            zip(self._endpoint_tiles(), ports, self._channels)
        ):
            symbol = self._end_symbol(end)
            _route_endpoint_op(
                symbol,
                tile.op,
                port,
                channel_index=channel,
                fifo_name=symbol if end in allocated else None,
                loc=loc,
                ip=ip,
            )
        self._op = _route_op(
            self._end_symbol(0),
            [self._end_symbol(end) for end in range(1, len(self._channels))],
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


class PacketFlow(_Route):
    """An explicit packet-switched route from a source to one or more destinations.

    Connects ``(src_tile, src_port, src_channel)`` to each destination
    endpoint, tagging the stream with `pkt_id`. Lowers to a single
    `aie.packetflow` op holding one `aie.packet_source` and one
    `aie.packet_dest` per destination. The user is responsible for
    arranging matching [`TileDma`][iron.TileDma] channels on the producer and
    consumer ends, which [`endpoint`][iron.dataflow.flow.PacketFlow.endpoint]
    names. Its channels are always given: the compiler assigns channels only
    to a [`Flow`][iron.Flow].
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
        name: str | None = None,
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
            src_channel: Source channel.  Defaults to 0.
            dst_port: Destination port bundle (as for [`Flow`][iron.Flow]).
            dst_channel: Destination channel.  Defaults to 0.
            extra_dsts: Additional destination endpoints if this packet needs
                to fan out. Each is a [`PacketDest`][iron.PacketDest].
            keep_pkt_header: If `True`, downstream tile receives the 4-byte
                packet header alongside the payload (useful when the receiver
                needs to re-emit with the same pkt_id). Defaults to `False`.
            shim_symbol: Same meaning as on [`Flow`][iron.Flow] — emit a
                matching `aie.shim_dma_allocation` of that name at the shim
                source, else at the one shim destination.
            name: Same meaning as on [`Flow`][iron.Flow]; left ``None``, it is
                named ``packetflow{n}``.
        """
        self._site = SourceSite.capture()
        self._pkt_id = pkt_id
        self._src = src
        self._dsts = [dst, *(d.tile for d in extra_dsts)]
        self._src_port = src_port
        self._dst_port = dst_port
        self._channels = [src_channel, dst_channel, *(d.channel for d in extra_dsts)]
        self._extra_dsts: list[PacketDest] = list(extra_dsts)
        self._endpoints: dict[int, FlowEndpoint] = {}
        self._keep_pkt_header = keep_pkt_header
        self._shim_symbol = shim_symbol
        self._shim_ends: set[int] = set()
        self._name = name
        self._op = None

    @property
    def pkt_id(self) -> int:
        return self._pkt_id

    def fill(self, source, pkt_type: int = 0, **kwargs):
        """Send data from the ``source`` runtime buffer into this route.

        Call from within a [`Runtime`][iron.Runtime] sequence body, on a
        PacketFlow whose src is a shim tile. The shim stamps every packet with
        ``pkt_type`` and this route's ``pkt_id``, so several PacketFlows leaving
        one shim channel each fill their own route. See ``emit_shim_transfer``
        for the keyword arguments; returns a
        [`Task`][iron.runtime.dmataskhandle.Task] handle to the transfer.
        """
        return self._transfer(
            source, DMAChannelDir.MM2S, packet=(pkt_type, self._pkt_id), **kwargs
        )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is not None:
            return
        loc = loc or self._site.location(self._name)
        dests = [
            {
                "dest": self._dsts[0].op,
                "port": self._dst_port,
                "channel": self._channels[1],
            }
        ]
        for d in self._extra_dsts:
            dests.append({"dest": d.tile.op, "port": d.port, "channel": d.channel})
        self._op = _packetflow_op(
            pkt_id=self._pkt_id,
            source=self._src.op,
            source_port=self._src_port,
            source_channel=self._channels[0],
            dests=dests,
            keep_pkt_header=self._keep_pkt_header,
            loc=loc,
            ip=ip,
        )
        if self._shim_symbol is not None or self._shim_ends:
            self._emit_shim_dma_alloc(loc, ip)
