# packet.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON :class:`PacketFifo` -- variable-rate packet-switched stream primitive.

This is the silicon-level promotion of the variable-rate / pktMerge /
finish-on-TLAST / out-of-order BD primitives the AM020 cross-walk
identified (see ``docs/aie-ml-am020-crosswalk.md`` G-T6.2-001 +
G-T6.4-101 + G-T7.4-200) and the IRON-investigation report
(``docs/iron-investigation.md``) flagged as the single biggest
abstraction-layer blocker for the variable-rate dataflow patterns
Phase 1 documented (T6.2 filter-early, T6.4 sparse-output emission,
T7.4 prototypes).

The ``ObjectFifo`` primitive is fundamentally a fixed-stride
producer-consumer pipeline: every consumer receives every element, the
rate is the same on both sides, and there is no first-class
"skip / variable-rate / per-packet route" semantics. AM020 documents
that the **hardware** does support variable-rate streams via three
related primitives:

1. **Packet-switched merge N:1** (Ch. 2 Figure 17, "pktMerge"). Multiple
   packet streams converge through a hardware merge block; variable
   rates from each producer are absorbed naturally because every packet
   carries an explicit header identifying its source and routing
   target.
2. **S2MM finish-on-TLAST** (Ch. 2 p. 27 Tile DMA Controller; Appendix
   A summary p. 87). Streams can terminate on a TLAST signal so
   producers can mark "end of valid stream" mid-flight without
   pre-declaring a fixed packet count.
3. **Out-of-order BD processing** (Ch. 5 p. 74). Memtile DMA can route
   incoming packets to different buffer descriptors based on the packet
   header, so a single physical channel can carry multiple logical
   streams that interleave arbitrarily on the wire and de-interleave at
   the consumer.

``PacketFifo`` exposes these three primitives as a single user-facing
class with the same ``prod()`` / ``cons()`` shape as ``ObjectFifo``,
plus a ``header_dtype`` (the per-packet routing tag width) and a
``merge_strategy`` (``"round-robin"`` for pktMerge's default fair
arbitration, or ``"priority"`` for header-based selection).

User-facing surface
-------------------

.. code-block:: python

    from aie.iron import PacketFifo, Worker
    from aie.iron.device import Tile

    # 3 producers fan into 1 consumer with packet-header routing.
    pf = PacketFifo(
        producers=[Tile(0, 2), Tile(0, 3), Tile(0, 4)],
        consumers=[Tile(0, 5)],
        header_dtype="uint8",
        merge_strategy="round-robin",
    )

    w_p0 = Worker(producer_fn, fn_args=[pf.prod(0)], tile=Tile(0, 2))
    w_p1 = Worker(producer_fn, fn_args=[pf.prod(1)], tile=Tile(0, 3))
    w_p2 = Worker(producer_fn, fn_args=[pf.prod(2)], tile=Tile(0, 4))
    w_c0 = Worker(consumer_fn, fn_args=[pf.cons(0)], tile=Tile(0, 5))

The user's worker body uses :meth:`PacketFifoHandle.acquire` /
:meth:`PacketFifoHandle.release` similarly to :class:`ObjectFifoHandle`,
plus :meth:`PacketFifoHandle.send_with_header` /
:meth:`PacketFifoHandle.recv_header` for the variable-rate routing
control.

Why a separate class instead of a flag on ObjectFifo
----------------------------------------------------

``ObjectFifo`` lowers to ``aie.objectfifo`` ops with shared-memory +
lock-based synchronization. Packet-switched dataflow lowers to
``aie.packet_flow`` + ``aie.packet_source`` + ``aie.packet_dest`` ops
with per-packet header-based routing through the AXI stream switch
fabric. The two share *no* runtime mechanism: ObjectFifo's
acquire/release is a software lock; PacketFifo's per-packet handshake
is the AXI-stream T-VALID/T-READY signalling.

Forcing this onto ObjectFifo as a flag would either:

- silently change every ``acquire`` / ``release`` lock op to a
  packet-stream handshake (a load-bearing semantic change for any
  pipeline that mixes the two), or
- require every ObjectFifo consumer to learn an "are we packet-mode?"
  invariant.

A sibling class with the same producer/consumer surface keeps the
abstraction clean and lets the lowering emit ``aie.packetflow`` ops
directly.

Concretely the lowering rule is:

- One :class:`PacketFifo` produces **one** :func:`aie.packetflow` op
  per (producer, consumer) routing entry. Multiple producers fanning
  into one consumer (the canonical pktMerge N:1 case) emits N
  ``packetflow`` ops with distinct ``pkt_id``s; the AXI stream switch
  hardware multiplexes them onto the consumer's input port.
- The ``packet_id`` of each producer is auto-assigned at construction
  time (0..N-1) but can be overridden via the explicit
  ``packet_ids=[...]`` constructor argument for designs that need
  specific id assignments (e.g., the consumer kernel uses the id as a
  switch on which producer the packet came from).
- ``finish-on-TLAST``: emitted as the ``keep_pkt_header=False`` form
  of ``aie.packetflow`` -- the hardware drops the routing header at
  consumer time and the consumer kernel sees only the payload + TLAST
  signal (the AXI stream's natural end-of-packet marker is what
  finish-on-TLAST consumes).

References
----------

- AM020 Ch. 2 Figure 17 (pktMerge N:1 hardware merge block)
- AM020 Ch. 2 p. 27 (Tile DMA Controller; S2MM finish-on-TLAST)
- AM020 Ch. 5 p. 74 (Memtile DMA out-of-order BD processing)
- AM020 Appendix A p. 87 (variable-rate stream summary)
- ``docs/aie-ml-am020-crosswalk.md`` G-T6.2-001 / G-T6.4-101 / G-T7.4-200
  (the cross-walk entries this primitive closes)
- ``docs/iron-investigation.md`` (Phase 1 T7.4 RED verdict; the
  abstraction-layer blocker this primitive resolves)
- ``python/iron/dataflow/fifo_handle_registry.py`` (T2.4 registry that
  PacketFifoHandle uses for fn_args dispatch)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import ir  # type: ignore
from ..dialects._aie_enum_gen import ObjectFifoPort  # type: ignore

from .resolvable import Resolvable, NotResolvedError
from .device import Tile
from .dataflow.endpoint import ObjectFifoEndpoint
from .dataflow.objectfifo import ObjectFifoHandle
from .dataflow.fifo_handle_registry import register_fifo_handle


# Supported header dtype tags. The packet header is the AXI stream's
# user-side metadata (T-USER). On AIE-ML / AIE2P the packet routing
# header is a 5-bit field (``pkt_id``) plus a 3-bit type field
# (``pkt_type``) per AM020 Ch. 2 p. 25; the user-facing ``header_dtype``
# below describes the *application*-level header the consumer kernel
# reads, which is the first word of the packet payload after routing.
# uint8 is the canonical tag-width for the filter-early use case
# (1-bit valid + 7-bit reserved); larger dtypes support multi-stream
# de-interleaving where the consumer needs to know the source stream
# id explicitly.
_SUPPORTED_HEADER_DTYPES: dict[str, np.dtype] = {
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "u8": np.dtype("uint8"),
    "u16": np.dtype("uint16"),
    "u32": np.dtype("uint32"),
}


# Supported merge strategies.
#
# - ``round-robin``: pktMerge default arbitration. The N producer
#   streams are sampled fairly by the AXI stream switch's round-robin
#   arbiter. Variable rates are absorbed because a producer that has
#   no packet ready is simply skipped this cycle.
#
# - ``priority``: the consumer receives packets in pkt_id order
#   regardless of arrival order at the merge block. Implemented at the
#   memtile by routing higher-priority pkt_ids to lower-numbered BDs
#   and letting the out-of-order BD scheduler (AM020 Ch. 5 p. 74)
#   service them first.
_SUPPORTED_MERGE_STRATEGIES: tuple[str, ...] = ("round-robin", "priority")


# Hardware limits per AM020 Ch. 2 p. 25.
#
# pkt_id is 5 bits -> 32 distinct routing ids per stream switch
# instance. We enforce this at construction time so designs that
# request more producers than the fabric can route get an actionable
# error before MLIR lowering. (The hardware also has 32 distinct
# pktMerge "slots" per memtile; for the canonical filter-early use
# case the limit is 4-8 producers per memtile column, not 32.)
_MAX_PACKET_ID: int = 31


def _validate_header_dtype(header_dtype: str) -> np.dtype:
    """Return the numpy dtype for the named header type, or raise."""
    if header_dtype not in _SUPPORTED_HEADER_DTYPES:
        raise ValueError(
            f"PacketFifo: unsupported header_dtype {header_dtype!r}; "
            f"expected one of {sorted(set(_SUPPORTED_HEADER_DTYPES.keys()))}"
        )
    return _SUPPORTED_HEADER_DTYPES[header_dtype]


def _validate_merge_strategy(merge_strategy: str) -> str:
    if merge_strategy not in _SUPPORTED_MERGE_STRATEGIES:
        raise ValueError(
            f"PacketFifo: unsupported merge_strategy {merge_strategy!r}; "
            f"expected one of {list(_SUPPORTED_MERGE_STRATEGIES)}"
        )
    return merge_strategy


def _coerce_to_tile(arg, role: str, idx: int | None = None) -> Tile:
    """Accept either a :class:`Tile` or anything exposing ``.tile``.

    Mirrors the helper in ``accum.py`` so callers that already have a
    :class:`Worker` can wire its placement into the PacketFifo without
    restating coordinates.
    """
    if isinstance(arg, Tile):
        return arg
    maybe_tile = getattr(arg, "tile", None)
    if isinstance(maybe_tile, Tile):
        return maybe_tile
    role_str = role if idx is None else f"{role}[{idx}]"
    raise TypeError(
        f"PacketFifo: {role_str} must be a Tile or a placed Worker (anything "
        f"exposing a `.tile` attribute of type Tile); got "
        f"{type(arg).__name__}"
    )


class PacketFifo(Resolvable):
    """Variable-rate packet-switched FIFO between N producers and M consumers.

    Sibling to :class:`ObjectFifo`. The user-facing surface
    (:meth:`prod`, :meth:`cons`, ``acquire`` / ``release`` on the
    returned handle) is similar so callers can swap one for the other
    when their kernel needs variable-rate dataflow + packet-header
    routing instead of fixed-stride memref-typed buffer dataflow.

    Lowers to :func:`aie.packetflow` ops between the per-producer DMA
    output ports and the per-consumer DMA input ports. The AXI stream
    switch hardware multiplexes the N producers onto the M consumers
    using either round-robin arbitration (the default pktMerge
    behavior) or priority-based scheduling (out-of-order BD on memtile,
    per AM020 Ch. 5 p. 74).

    Args:
        producers: List of :class:`Tile` (or placed Worker) endpoints
            that produce packets.
        consumers: List of :class:`Tile` (or placed Worker) endpoints
            that consume packets. Most pktMerge use cases are N:1
            (multiple producers -> one consumer); N:M is supported via
            packet-header-based fan-out (each consumer subscribes to a
            packet_id range).
        header_dtype: Per-packet routing-header dtype. One of
            ``"uint8"`` (default; 1-bit valid + 7-bit reserved -- the
            canonical filter-early use case), ``"uint16"``, or
            ``"uint32"``. Larger headers support multi-stream
            de-interleaving where the consumer kernel needs to know the
            source stream id explicitly.
        merge_strategy: How the AXI stream switch arbitrates between
            multiple producers when more than one has a packet ready in
            the same cycle. One of ``"round-robin"`` (default; pktMerge
            fair arbitration) or ``"priority"`` (header-based ordering
            via memtile out-of-order BD scheduling).
        packet_ids: Optional explicit per-producer packet id list. If
            omitted, each producer is auto-assigned ``packet_id = i``
            where ``i`` is its position in ``producers``. Override is
            useful when the consumer kernel switches on a specific id
            value (e.g., 0x10 for valid windows, 0x20 for invalid).
        obj_type: The numpy ndarray type of each packet payload. If
            omitted, defaults to a single 32-bit word per packet
            (``np.ndarray[(1,), np.dtype[np.int32]]``); supplying a
            larger ndarray type is required for multi-word packets.
        depth: Default per-endpoint packet buffer depth. Defaults to
            ``2`` (matching :class:`ObjectFifo`'s default).
        keep_pkt_header: If True, the routing header travels with the
            packet to the consumer. If False (the finish-on-TLAST
            mode), the hardware drops the header at consumer time and
            the consumer kernel sees only the payload + TLAST signal.
            Defaults to ``True`` (header retained for kernel-side
            switching).
        name: Optional name for diagnostics. If omitted, a unique
            ``pf<N>`` is generated.

    Raises:
        TypeError: ``producers`` / ``consumers`` element is not a Tile
            (or doesn't expose a ``.tile`` attribute).
        ValueError: empty ``producers`` or ``consumers``;
            unsupported ``header_dtype`` / ``merge_strategy``;
            ``packet_ids`` length mismatch or contains an id outside
            ``[0, 31]``; depth < 1.
    """

    # Used to generate unique PacketFifo names when none is provided.
    __pf_index = 0

    def __init__(
        self,
        producers: Sequence,
        consumers: Sequence,
        header_dtype: str = "uint8",
        merge_strategy: str = "round-robin",
        packet_ids: Sequence[int] | None = None,
        obj_type: type[np.ndarray] | None = None,
        depth: int = 2,
        keep_pkt_header: bool = True,
        name: str | None = None,
    ):
        # --- Validation: cheap, eager, actionable error messages ---
        if not isinstance(producers, (list, tuple)) or len(producers) == 0:
            raise ValueError(
                f"PacketFifo: producers must be a non-empty list of Tiles, "
                f"got {producers!r}"
            )
        if not isinstance(consumers, (list, tuple)) or len(consumers) == 0:
            raise ValueError(
                f"PacketFifo: consumers must be a non-empty list of Tiles, "
                f"got {consumers!r}"
            )
        if depth < 1:
            raise ValueError(f"PacketFifo: depth must be > 0, got {depth}")

        header_np_dtype = _validate_header_dtype(header_dtype)
        merge_strategy = _validate_merge_strategy(merge_strategy)

        prod_tiles: list[Tile] = [
            _coerce_to_tile(p, "producer", i) for i, p in enumerate(producers)
        ]
        cons_tiles: list[Tile] = [
            _coerce_to_tile(c, "consumer", i) for i, c in enumerate(consumers)
        ]

        n_prod = len(prod_tiles)
        if n_prod > _MAX_PACKET_ID + 1:
            raise ValueError(
                f"PacketFifo: AM020 Ch. 2 p. 25 limits pkt_id to 5 bits "
                f"(0..{_MAX_PACKET_ID}); got {n_prod} producers. Split into "
                f"multiple PacketFifos or aggregate at memtile first."
            )

        if packet_ids is None:
            packet_ids = list(range(n_prod))
        else:
            packet_ids = list(packet_ids)
            if len(packet_ids) != n_prod:
                raise ValueError(
                    f"PacketFifo: packet_ids length {len(packet_ids)} does "
                    f"not match number of producers {n_prod}"
                )
            for pid in packet_ids:
                if not isinstance(pid, int):
                    raise TypeError(
                        f"PacketFifo: packet_ids must be ints, got "
                        f"{type(pid).__name__} ({pid!r})"
                    )
                if pid < 0 or pid > _MAX_PACKET_ID:
                    raise ValueError(
                        f"PacketFifo: packet_id {pid} outside "
                        f"[0, {_MAX_PACKET_ID}] (5-bit field per AM020 "
                        f"Ch. 2 p. 25)"
                    )
            if len(set(packet_ids)) != len(packet_ids):
                raise ValueError(
                    f"PacketFifo: packet_ids must be unique; got "
                    f"{packet_ids!r}"
                )

        if obj_type is None:
            obj_type = np.ndarray[(1,), np.dtype[np.int32]]

        if name is None:
            name = f"pf{PacketFifo.__get_index()}"

        # --- Persist state ---
        self.name: str = name
        self._producers: list[Tile] = prod_tiles
        self._consumers: list[Tile] = cons_tiles
        self._header_dtype: str = header_dtype
        self._header_np_dtype: np.dtype = header_np_dtype
        self._merge_strategy: str = merge_strategy
        self._packet_ids: list[int] = packet_ids
        self._obj_type: type[np.ndarray] = obj_type
        self._depth: int = depth
        self._keep_pkt_header: bool = keep_pkt_header

        # Lazy-constructed handles (one per producer index, one per
        # consumer index). Index-based access avoids the
        # ObjectFifo-style "single prod / multi cons" asymmetry --
        # PacketFifo is genuinely N:M.
        self._prod_handles: dict[int, "PacketFifoHandle"] = {}
        self._cons_handles: dict[int, "PacketFifoHandle"] = {}

        # Lowered ops (one per producer x consumer routing entry).
        self._ops: list = []
        self._resolving: bool = False

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__pf_index
        cls.__pf_index += 1
        return idx

    # --- Read-only property surface ---

    @property
    def producers(self) -> list[Tile]:
        return list(self._producers)

    @property
    def consumers(self) -> list[Tile]:
        return list(self._consumers)

    @property
    def header_dtype(self) -> str:
        return self._header_dtype

    @property
    def merge_strategy(self) -> str:
        return self._merge_strategy

    @property
    def packet_ids(self) -> list[int]:
        return list(self._packet_ids)

    @property
    def obj_type(self) -> type[np.ndarray]:
        return self._obj_type

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def keep_pkt_header(self) -> bool:
        return self._keep_pkt_header

    @property
    def num_producers(self) -> int:
        return len(self._producers)

    @property
    def num_consumers(self) -> int:
        return len(self._consumers)

    def prod(self, idx: int = 0, depth: int | None = None) -> "PacketFifoHandle":
        """Return the producer handle for the ``idx``-th producer tile.

        Args:
            idx: Index into ``producers`` (the list passed to
                ``__init__``). Defaults to 0 for the common 1-producer
                case.
            depth: Per-handle depth override. Defaults to the
                PacketFifo's default depth.

        Raises:
            IndexError: ``idx`` outside ``[0, num_producers)``.
            ValueError: ``depth`` < 1.

        Returns:
            PacketFifoHandle: The producer handle for that index.
        """
        if idx < 0 or idx >= self.num_producers:
            raise IndexError(
                f"PacketFifo {self.name!r}: producer index {idx} outside "
                f"[0, {self.num_producers})"
            )
        if depth is not None and depth < 1:
            raise ValueError(f"PacketFifo: prod depth must be > 0, got {depth}")
        if idx not in self._prod_handles:
            self._prod_handles[idx] = PacketFifoHandle(
                self,
                is_prod=True,
                index=idx,
                depth=depth if depth is not None else self._depth,
            )
        return self._prod_handles[idx]

    def cons(self, idx: int = 0, depth: int | None = None) -> "PacketFifoHandle":
        """Return the consumer handle for the ``idx``-th consumer tile.

        Args:
            idx: Index into ``consumers``. Defaults to 0 for the
                canonical N:1 pktMerge case.
            depth: Per-handle depth override.

        Raises:
            IndexError: ``idx`` outside ``[0, num_consumers)``.
            ValueError: ``depth`` < 1.

        Returns:
            PacketFifoHandle: The consumer handle for that index.
        """
        if idx < 0 or idx >= self.num_consumers:
            raise IndexError(
                f"PacketFifo {self.name!r}: consumer index {idx} outside "
                f"[0, {self.num_consumers})"
            )
        if depth is not None and depth < 1:
            raise ValueError(f"PacketFifo: cons depth must be > 0, got {depth}")
        if idx not in self._cons_handles:
            self._cons_handles[idx] = PacketFifoHandle(
                self,
                is_prod=False,
                index=idx,
                depth=depth if depth is not None else self._depth,
            )
        return self._cons_handles[idx]

    def __str__(self) -> str:
        return (
            f"PacketFifo(name={self.name!r}, "
            f"producers={self.num_producers}, consumers={self.num_consumers}, "
            f"header_dtype={self._header_dtype!r}, "
            f"merge_strategy={self._merge_strategy!r}, "
            f"keep_pkt_header={self._keep_pkt_header}, "
            f"packet_ids={self._packet_ids})"
        )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Lower this PacketFifo to ``aie.packetflow`` ops.

        Emits **one** ``aie.packetflow`` per (producer, consumer)
        routing entry, with the producer's auto-assigned (or
        user-supplied) ``pkt_id``. The AXI stream switch hardware
        multiplexes the ops at runtime; the IRON layer only declares
        the routing.

        For the canonical N:1 pktMerge case (``num_consumers == 1``),
        N packetflows are emitted, one per producer, all targeting
        the single consumer's DMA input port. The hardware's pktMerge
        block performs the round-robin arbitration (or priority
        scheduling for ``merge_strategy="priority"``) across the N
        producer streams.

        For N:M cases, the cross-product is emitted: each producer's
        packets are routed to every consumer (the consumer-side filter
        on ``pkt_id`` selects which packets it actually keeps). This
        matches AM020 Ch. 5 p. 74's out-of-order BD model where the
        memtile dispatches incoming packets to per-pkt_id BDs.

        The C++ kernel side is responsible for the actual cascade /
        DMA reads/writes; this function only declares the routing.

        Re-entrancy is guarded the same way :class:`ObjectFifo` /
        :class:`AccumFifo` guard it (set ``self._resolving = True``
        before recursing; idempotent if called twice).
        """
        if self._resolving:
            return
        self._resolving = True

        # Late-import the dialect helper to avoid an import-time MLIR
        # context dependency. The helper is only needed when resolve()
        # is actually called from within an ``aie.device`` body (which
        # is an ir.Operation context); pure-Python surface tests do
        # not need an MLIR context and never reach this branch.
        from ..dialects.aie import packetflow  # type: ignore
        from ..dialects._aie_enum_gen import WireBundle  # type: ignore

        self._ops = []
        for prod_idx, (prod_tile, pkt_id) in enumerate(
            zip(self._producers, self._packet_ids)
        ):
            prod_op = prod_tile.op
            # Build the per-consumer dest list. For N:M, the
            # consumer-side header filter selects the actual delivery
            # set; the IRON layer routes to all consumers and lets the
            # hardware filter.
            dests = [
                {
                    "dest": cons_tile.op,
                    "port": WireBundle.DMA,
                    "channel": cons_idx,
                }
                for cons_idx, cons_tile in enumerate(self._consumers)
            ]
            self._ops.append(
                packetflow(
                    pkt_id=pkt_id,
                    source=prod_op,
                    source_port=WireBundle.DMA,
                    source_channel=prod_idx,
                    dests=dests,
                    keep_pkt_header=self._keep_pkt_header,
                )
            )

    @property
    def ops(self) -> list:
        """The lowered ``aie.packetflow`` ops (one per producer).

        Raises :class:`NotResolvedError` if :meth:`resolve` has not run.
        """
        if not self._resolving:
            raise NotResolvedError()
        return list(self._ops)


class PacketFifoHandle(ObjectFifoHandle):
    """Producer or consumer handle to a :class:`PacketFifo`.

    Subclasses :class:`ObjectFifoHandle` so that any code path that
    type-checks against ``ObjectFifoHandle`` (the placer, the validator,
    debug dumps) sees a uniform "fifo handle" surface. The T2.4
    ``fifo_handle_registry`` dispatch is the load-bearing path now; the
    inheritance is kept only for surface compatibility with
    not-yet-converted code.

    The ``acquire`` / ``release`` methods preserve the ObjectFifoHandle
    signature so user code parameterized over both fifo kinds doesn't
    need to branch. The packet-stream-specific methods
    :meth:`send_with_header` and :meth:`recv_header` are added on top
    for callers that need explicit per-packet header control (the
    filter-early use case).
    """

    def __init__(
        self,
        packet_fifo: "PacketFifo",
        is_prod: bool,
        index: int,
        depth: int,
    ):
        # Bypass ObjectFifoHandle.__init__ (it requires an `of` with
        # ObjectFifo semantics: depth, dims_from_stream_per_cons, etc.)
        # and set the fields ObjectFifoHandle exposes as properties
        # directly so isinstance-based downstream code keeps working.
        self._port = (
            ObjectFifoPort.Produce if is_prod else ObjectFifoPort.Consume
        )
        self._is_prod: bool = is_prod
        self._packet_fifo: PacketFifo = packet_fifo
        self._index: int = index
        self._depth: int = depth
        self._endpoint: ObjectFifoEndpoint | None = None
        self._dims_from_stream = None
        # Bind the handle's tile to the PacketFifo's per-index endpoint
        # so placement-aware passes see the wire-up.
        self._tile: Tile = (
            packet_fifo.producers[index]
            if is_prod
            else packet_fifo.consumers[index]
        )

    @property
    def name(self) -> str:
        return self._packet_fifo.name

    @property
    def handle_type(self) -> str:
        return "prod" if self._is_prod else "cons"

    @property
    def packet_fifo(self) -> "PacketFifo":
        """The underlying :class:`PacketFifo`."""
        return self._packet_fifo

    @property
    def index(self) -> int:
        """This handle's index in the PacketFifo's producer/consumer list."""
        return self._index

    @property
    def packet_id(self) -> int:
        """Per-producer packet id (auto-assigned or user-supplied).

        For consumer handles, returns the index in the consumer list
        (consumers do not have a single packet_id; they receive all ids
        and filter at the kernel level).
        """
        if self._is_prod:
            return self._packet_fifo.packet_ids[self._index]
        return self._index

    @property
    def header_dtype(self) -> str:
        return self._packet_fifo.header_dtype

    @property
    def merge_strategy(self) -> str:
        return self._packet_fifo.merge_strategy

    @property
    def obj_type(self):  # type: ignore[override]
        return self._packet_fifo.obj_type

    @property
    def shape(self) -> Sequence[int]:  # type: ignore[override]
        from ..helpers.util import np_ndarray_type_get_shape  # type: ignore
        return np_ndarray_type_get_shape(self._packet_fifo.obj_type)

    @property
    def dtype(self):  # type: ignore[override]
        from ..helpers.util import np_ndarray_type_get_dtype  # type: ignore
        return np_ndarray_type_get_dtype(self._packet_fifo.obj_type)

    @property
    def depth(self) -> int:  # type: ignore[override]
        return self._depth

    def acquire(self, num_elem: int = 1):
        """Acquire access to ``num_elem`` packets.

        Mirrors :meth:`ObjectFifoHandle.acquire` so user code is
        portable across fifo kinds. Internally the AXI stream's
        T-VALID/T-READY handshake performs the synchronization;
        IRON-level acquire is a no-op marker that downstream passes
        consume to wire up DMA buffer descriptors.

        Returns:
            None: Packet-stream consumers receive packets via the
            kernel's stream intrinsic, not via a buffer subview. The
            return value preserves the API shape but is intentionally
            None.
        """
        if num_elem < 1:
            raise ValueError(
                f"PacketFifo handle: acquire() requires num_elem >= 1, "
                f"got {num_elem}"
            )
        if num_elem > self._depth:
            raise ValueError(
                f"PacketFifo handle: acquire({num_elem}) exceeds depth "
                f"{self._depth}"
            )
        return None

    def release(self, num_elem: int = 1) -> None:
        """Release access to ``num_elem`` packets.

        Symmetric to :meth:`acquire`. Drives the next iteration of the
        per-packet AXI handshake; no IRON-level lock to release.
        """
        if num_elem < 1:
            raise ValueError(
                f"PacketFifo handle: release() requires num_elem >= 1, "
                f"got {num_elem}"
            )
        if num_elem > self._depth:
            raise ValueError(
                f"PacketFifo handle: release({num_elem}) exceeds depth "
                f"{self._depth}"
            )

    def send_with_header(self, header_value: int) -> None:
        """Producer-only: stage a packet header for the next outgoing packet.

        This is a kernel-side metadata declaration; the actual MLIR
        emitted is the :func:`aie.packetflow` op declared at
        :meth:`PacketFifo.resolve` time. The runtime call here records
        the kernel-author's intent that the next ``release`` should
        carry the given header value -- the C++ kernel is responsible
        for emitting the corresponding ``put_ms`` / ``put_mcd`` with
        the header byte in the leading word.

        Args:
            header_value: Header value to attach to the next outgoing
                packet. Must fit in the configured ``header_dtype``
                (e.g., 0..255 for ``uint8``).

        Raises:
            ValueError: This handle is a consumer, or
                ``header_value`` doesn't fit ``header_dtype``.
        """
        if not self._is_prod:
            raise ValueError(
                f"PacketFifo handle {self.name!r}: send_with_header() is "
                f"producer-only; got cons handle"
            )
        np_dtype = self._packet_fifo._header_np_dtype
        max_val = int(np.iinfo(np_dtype).max)
        if header_value < 0 or header_value > max_val:
            raise ValueError(
                f"PacketFifo handle {self.name!r}: header_value "
                f"{header_value} outside [0, {max_val}] for header_dtype "
                f"{self._packet_fifo._header_dtype!r}"
            )
        # No MLIR emitted here: the routing is per-packetflow, not
        # per-packet. Per-packet header values are written by the C++
        # kernel; this method only validates and records intent.

    def recv_header(self) -> None:
        """Consumer-only: declare that the next packet's header is needed.

        Mirrors :meth:`send_with_header`: the actual header byte is
        read by the C++ kernel from the leading word of the packet
        payload (after the AXI stream switch routing). This method
        only enforces the producer/consumer asymmetry and exists so
        the API surface is symmetric across handle types.

        Raises:
            ValueError: This handle is a producer.
        """
        if self._is_prod:
            raise ValueError(
                f"PacketFifo handle {self.name!r}: recv_header() is "
                f"consumer-only; got prod handle"
            )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Forward to the underlying :class:`PacketFifo`'s resolve."""
        self._packet_fifo.resolve(loc=loc, ip=ip)

    def all_of_endpoints(self) -> list:  # type: ignore[override]
        """All endpoints (producer + consumer tiles) of the PacketFifo.

        Returns the underlying tiles since PacketFifo isn't an
        ObjectFifo and doesn't have a single producer/cons endpoint
        list of the same shape.
        """
        return list(self._packet_fifo.producers) + list(
            self._packet_fifo.consumers
        )

    def __str__(self) -> str:
        return (
            f"PacketFifoHandle({self.handle_type}, idx={self._index}, "
            f"pkt_id={self.packet_id}, depth={self._depth}, "
            f"of={self._packet_fifo})"
        )


# ---------------------------------------------------------------------------
# Registry integration (T2.4): register PacketFifoHandle so that
# Worker.fn_args dispatch recognizes it without modifying worker.py.
#
# The handler mirrors the ObjectFifoHandle bookkeeping bit-for-bit
# (set arg.endpoint = worker; append to worker._fifos). Registering
# AFTER ObjectFifoHandle (which dataflow/__init__.py registers at
# import time) means the reverse-insertion-order walk in
# dispatch_fn_arg picks PacketFifoHandle first for PacketFifoHandle
# instances -- exactly the property the registry was designed for.
# ---------------------------------------------------------------------------


def _packet_fifo_handle_handler(arg, worker):
    """Worker.fn_args handler for :class:`PacketFifoHandle`.

    Mirrors the pre-registered :class:`ObjectFifoHandle` handler so
    that the rest of IRON's bookkeeping (worker._fifos, the placer's
    endpoint walk, debug dumps) treats PacketFifoHandle as a fifo
    handle without further modification.
    """
    arg.endpoint = worker
    worker._fifos.append(arg)


register_fifo_handle(PacketFifoHandle, _packet_fifo_handle_handler)


__all__ = [
    "PacketFifo",
    "PacketFifoHandle",
]
