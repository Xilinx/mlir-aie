# accum.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON :class:`AccumFifo` -- FP32 accumulator inter-tile / inter-timestep state.

A first-class IRON primitive over two AM020-documented hardware paths
for FP32 accumulator-register state passing. Surfaces both as a single
dataflow primitive:

1. **BM-to-BM register move** (intra-tile, across timesteps):
   AM020 Ch. 4 p. 67 -- "Accumulator to accumulator: Move one 512-bit
   accumulator (AM) register to another AM-register in one cycle."
   The producer/consumer alias the same tile; the lowered MLIR keeps
   the accumulator state in BM registers across the worker's
   ``while(true)`` body, never narrowing to vector-register width.

2. **Cascade-stream BM transfer** (inter-tile):
   AM020 Ch. 4 p. 67 -- "Cascade stream connects the AIE-MLs in a chain
   and allows the AIE-MLs to transfer an accumulator register
   (512-bit) from one to the next."
   The producer and consumer are placed on different (typically
   vertically-adjacent) :class:`Tile` instances; the lowered MLIR emits
   :func:`aie.cascade_flow` between them.

User-facing surface
-------------------

.. code-block:: python

    from aie.iron import AccumFifo, Worker, ObjectFifo

    # Intra-tile (BM-to-BM register move): producer tile == consumer tile.
    # Use case: LSTM h/c persistence across timesteps within a single tile.
    h_state = AccumFifo(producer=tile_0_2, consumer=tile_0_2,
                        dtype="accfloat", lanes=16, name="h_state")

    # Inter-tile (cascade stream): producer / consumer on different tiles.
    # Use case: layer-N -> layer-N+1 accumulator hand-off in an LSTM stack.
    h_chain = AccumFifo(producer=tile_0_2, consumer=tile_0_3,
                        dtype="accfloat", lanes=16)

    w_first  = Worker(first_layer_fn,  fn_args=[h_chain.prod()])
    w_second = Worker(second_layer_fn, fn_args=[h_chain.cons()])

The user's worker body uses
:meth:`AccumFifoHandle.acquire` / :meth:`AccumFifoHandle.release`
identically to :class:`ObjectFifoHandle`. The lowered MLIR differs:

- A regular ``ObjectFifo`` lowers to ``aie.objectfifo`` with a
  ``memref<NxT>`` element type. The DMA write narrows the FP32
  accumulator to the buffer's element type at storage time -- the
  precision wall observed when LSTM-style recurrent state has to
  round-trip through a memref.
- An ``AccumFifo`` lowers to either a cascade-flow channel
  (``aie.cascade_flow`` between distinct producer/consumer tiles) or
  a same-tile in-register handoff (no DMA, no objectfifo memref) when
  producer and consumer alias the same tile. Either path preserves the
  full 23-bit accumulator mantissa.

Why a separate class instead of an ``ObjectFifo`` flag
------------------------------------------------------

``ObjectFifo`` is a memref-typed channel. The memref element type
constrains the storage width (bf16 / int32 / etc.) and the lowering
emits DMA buffer descriptors that copy memref words. The accumulator
register is **not** a memref word: it is a hardware register file slice
(the 512-bit "AM" register on AIE-ML / AIE2P), and its precision-
preserving transfers are *not* DMA copies. Modeling this as a flag on
``ObjectFifo`` would either force the memref type to be a fiction
(``memref<16xf32>`` that the lowering ignores) or require every
``ObjectFifo`` consumer to learn about an accumulator-mode invariant.
A sibling class with the same producer/consumer surface keeps the
abstraction clean and lets the lowering emit cascade-flow ops (or
in-register continuity, for same-tile producer/consumer) directly.

Concretely the lowering rule is:

- ``producer.tile == consumer.tile`` ->  no MLIR op emitted in
  :meth:`AccumFifo.resolve`. The intra-tile BM-to-BM register move is
  carried by the worker's C++ kernel (``aie::accum<accfloat,N>`` lives
  in registers; the kernel is the one that keeps the accumulator hot
  across the iteration boundary). The :class:`AccumFifoHandle`
  ``acquire`` / ``release`` calls are no-ops that exist purely to
  preserve dataflow-level invariants and let
  :class:`Worker.fn_args` keep the same shape regardless of whether
  the handle is intra-tile or cascade.

- ``producer.tile != consumer.tile`` -> :func:`aie.cascade_flow` is
  emitted between the two tiles. AM020 Appendix A p. 80 Figure 45
  (carried forward to AIE2P) confirms cascade routing is automatic
  for vertically-adjacent tiles in the same column. A vertical-
  adjacency check is enforced at construction time and an
  informational warning is raised for the horizontal case (the
  routing path is documented but less commonly exercised).

Notes on ``dtype``
------------------

The ``dtype`` argument is the AIE accumulator type tag, not a numpy
type. Supported values:

- ``"accfloat"`` -- FP32 accumulator (32-bit float, hardware-supported
  on AIE-ML / AIE2P; the FP32 cascade-LSTM path).
- ``"acc32"``    -- int32 accumulator.
- ``"acc64"``    -- int64 accumulator (paired-lane).

The default is ``"accfloat"`` since AccumFifo's primary motivating
use case (preserving accumulator precision across LSTM-style
recurrent state) is the FP32 accumulator path.

References
----------

- AM020 Ch. 4 p. 67 (Register Move Functionality + Cascade Stream)
- AM020 Ch. 4 p. 65 (FP32 accumulator width = 32-bit, 23 mantissa bits)
- AM020 Appendix A p. 80 Figure 45 (vertical+horizontal cascade grid)
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .. import ir  # type: ignore
from ..dialects._aie_enum_gen import AIETileType, ObjectFifoPort  # type: ignore
from ..dialects.aie import cascade_flow

from .resolvable import Resolvable, NotResolvedError
from .device import Tile
from .dataflow.endpoint import ObjectFifoEndpoint
from .dataflow.objectfifo import ObjectFifoHandle


# AM020 Ch. 4 p. 67: 512-bit cascade transfer = 16 lanes of int32
# OR 16 lanes of accfloat (FP32 accumulator). Carry-forward to AIE2P
# confirmed at the dialect level via aie.cascade_flow.
_CASCADE_BITS: int = 512

# Supported accumulator dtype tags. Mirrors aie_api/adf accumulator
# types (`accfloat`, `acc32`, `acc64`). `accfloat` is the FP32 path;
# `acc48` is AIE1-only (not on AIE-ML/AIE2P) and explicitly rejected
# with a clear error message rather than silently accepted.
_SUPPORTED_DTYPES: dict[str, int] = {
    "accfloat": 32,  # FP32 accumulator (AIE-ML / AIE2P)
    "acc32": 32,     # int32 accumulator
    "acc64": 64,     # int64 (paired-lane) accumulator
}


def _validate_dtype(dtype: str) -> int:
    """Return the per-lane bit-width of the named accumulator dtype, or raise.

    Raises ``ValueError`` for ``acc48`` (AIE1-only, not on AIE-ML/AIE2P)
    and for unrecognized tags. The check is by name -- numpy dtypes do
    not have AIE accumulator semantics so the API uses the AIE-side
    string tag directly.
    """
    if dtype == "acc48":
        raise ValueError(
            "AccumFifo: dtype='acc48' is AIE1-only and is not supported "
            "on AIE-ML / AIE2P (this is the device class IRON targets). "
            "Use 'accfloat' (FP32) or 'acc32' (int32) instead."
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"AccumFifo: unsupported dtype {dtype!r}; expected one of "
            f"{sorted(_SUPPORTED_DTYPES.keys())}"
        )
    return _SUPPORTED_DTYPES[dtype]


def _validate_lane_count(dtype: str, lanes: int) -> None:
    """Ensure the requested lane count fits in one 512-bit cascade transfer.

    AM020 Ch. 4 p. 67: cascade transfer is 512 bits / cycle.
    16 x f32 = 512; 16 x i32 = 512; 8 x i64 = 512.
    Lane counts that don't fit one cascade transfer (or are zero/negative)
    are rejected eagerly so the caller sees the failure at API-call time
    rather than during MLIR lowering.
    """
    if lanes <= 0:
        raise ValueError(f"AccumFifo: lanes must be > 0, got {lanes}")
    bits_per_lane = _SUPPORTED_DTYPES[dtype]
    total_bits = lanes * bits_per_lane
    if total_bits != _CASCADE_BITS:
        raise ValueError(
            f"AccumFifo: lanes={lanes} of dtype={dtype!r} ({bits_per_lane} "
            f"bits/lane) totals {total_bits} bits, but the AM020 Ch. 4 p. 67 "
            f"cascade transfer is exactly {_CASCADE_BITS} bits per cycle. "
            f"Use lanes=16 for accfloat/acc32 or lanes=8 for acc64."
        )


def _check_vertical_adjacency(producer: Tile, consumer: Tile) -> None:
    """Warn if producer/consumer aren't vertically adjacent in the same column.

    AM020 Appendix A p. 80 Figure 45 (carried forward to AIE2P) confirms
    cascade-stream routing exists for both vertical AND horizontal
    neighbours. The vertical case is the well-trodden path; we raise
    a :class:`UserWarning` for the less-exercised geometries
    (horizontal, diagonal, non-adjacent) rather than block the
    lowering, so callers can make an informed choice.
    """
    # Handle un-placed tiles (col/row may be None pre-placement-pass).
    if producer.col is None or consumer.col is None:
        return
    if producer.row is None or consumer.row is None:
        return

    same_column = producer.col == consumer.col
    row_delta = abs(consumer.row - producer.row)

    if same_column and row_delta == 1:
        return  # vertically adjacent -- the only case T7-IRON measured

    warnings.warn(
        f"AccumFifo: producer tile (col={producer.col}, row={producer.row}) "
        f"and consumer tile (col={consumer.col}, row={consumer.row}) are not "
        f"vertically adjacent in the same column. AM020 Appendix A p. 80 "
        f"Figure 45 documents both vertical and horizontal cascade routing, "
        f"but Phase 1's cascade_stream investigation only verified the "
        f"vertical case on AIE2P silicon. The lowering will still emit "
        f"aie.cascade_flow; expect placement-pass diagnostics if the "
        f"requested geometry isn't realisable on this device.",
        UserWarning,
        stacklevel=3,
    )


class AccumFifo(Resolvable):
    """FP32 (or int32 / int64) accumulator inter-tile / inter-timestep state.

    Sibling to :class:`ObjectFifo`. The user-facing surface
    (:meth:`prod`, :meth:`cons`, ``acquire`` / ``release`` on the
    returned handle) is identical so callers can swap one for the other
    when their kernel needs accumulator-precision state continuity
    instead of memref-typed buffer dataflow.

    Two physical lowerings, picked by tile placement:

    - **producer.tile == consumer.tile** (intra-tile BM-to-BM register
      move): no MLIR op emitted in :meth:`resolve`. The continuity is
      entirely the worker C++ kernel's job (an ``aie::accum<T,N>`` local
      that survives the worker's ``while(true)`` iteration boundary).
      The handle's ``acquire`` / ``release`` are no-ops that preserve
      the dataflow API shape so a chain of ``AccumFifo`` instances can
      be parameterized over both intra- and inter-tile cases without
      changing the worker body.

    - **producer.tile != consumer.tile** (cascade-stream BM transfer):
      :func:`aie.cascade_flow` is emitted between the two tiles in
      :meth:`resolve`. The C++ kernel uses ``put_mcd`` / ``get_scd``
      intrinsics to read/write the cascade port.

    Args:
        producer: The :class:`Tile` (or already-placed
            :class:`Worker` exposing ``.tile``) that produces the
            accumulator state.
        consumer: The :class:`Tile` (or worker) that consumes it.
            May be the same instance as ``producer`` for the
            intra-tile BM-to-BM register move case.
        dtype: AIE accumulator type tag. One of ``"accfloat"`` (FP32;
            default), ``"acc32"`` (int32), ``"acc64"`` (int64 paired-
            lane). ``"acc48"`` is rejected (AIE1-only, not on AIE-ML /
            AIE2P).
        lanes: Number of accumulator lanes per transfer. Must total
            exactly 512 bits (AM020 Ch. 4 p. 67 cascade transfer
            width): ``16`` for ``accfloat``/``acc32``, ``8`` for
            ``acc64``. Defaults to ``16`` (the FP32 case).
        name: Optional name for diagnostics. If omitted, a unique
            ``af<N>`` is generated.

    Raises:
        ValueError: ``dtype`` not supported, ``lanes`` doesn't total
            512 bits, or ``producer`` / ``consumer`` is not a
            :class:`Tile` (or doesn't expose a ``.tile`` attribute).
    """

    # Used to generate unique AccumFifo names when none is provided.
    __af_index = 0

    def __init__(
        self,
        producer,
        consumer,
        dtype: str = "accfloat",
        lanes: int = 16,
        name: str | None = None,
    ):
        # Validate dtype + lane width against AM020 Ch. 4 p. 67 first --
        # cheap and gives the user an actionable error before any tile
        # introspection.
        _validate_dtype(dtype)
        _validate_lane_count(dtype, lanes)

        prod_tile = self._coerce_to_tile(producer, "producer")
        cons_tile = self._coerce_to_tile(consumer, "consumer")

        # Intra-tile vs inter-tile is decided by tile identity (col/row),
        # not Python object identity -- callers may construct two Tile
        # objects with the same coordinates and we treat that as the
        # same tile (matches how the rest of IRON does it).
        same_tile = (
            prod_tile.col == cons_tile.col
            and prod_tile.row == cons_tile.row
        )

        if not same_tile:
            _check_vertical_adjacency(prod_tile, cons_tile)

        if name is None:
            name = f"af{AccumFifo.__get_index()}"

        self.name: str = name
        self._dtype: str = dtype
        self._lanes: int = lanes
        self._prod_tile: Tile = prod_tile
        self._cons_tile: Tile = cons_tile
        self._same_tile: bool = same_tile

        self._prod: AccumFifoHandle | None = None
        self._cons: AccumFifoHandle | None = None
        self._op = None  # cascade_flow op (if inter-tile); None if intra-tile
        self._resolving: bool = False

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__af_index
        cls.__af_index += 1
        return idx

    @staticmethod
    def _coerce_to_tile(arg, role: str) -> Tile:
        """Accept either a :class:`Tile` or anything exposing ``.tile``.

        The latter is the common case where the caller already has a
        :class:`Worker` and wants to wire its placement into the
        AccumFifo without restating the coordinates.
        """
        if isinstance(arg, Tile):
            return arg
        # Worker / ObjectFifoEndpoint expose `.tile`.
        maybe_tile = getattr(arg, "tile", None)
        if isinstance(maybe_tile, Tile):
            return maybe_tile
        raise ValueError(
            f"AccumFifo: {role} must be a Tile or a placed Worker (anything "
            f"exposing a `.tile` attribute of type Tile); got {type(arg).__name__}"
        )

    @property
    def dtype(self) -> str:
        """The accumulator dtype tag (``"accfloat"`` / ``"acc32"`` / ``"acc64"``)."""
        return self._dtype

    @property
    def lanes(self) -> int:
        """Number of accumulator lanes per cascade transfer."""
        return self._lanes

    @property
    def is_intra_tile(self) -> bool:
        """True if producer and consumer are placed on the same tile.

        Intra-tile AccumFifos lower to no MLIR ops -- the BM-to-BM register
        move is carried by the worker's C++ kernel keeping its
        ``aie::accum`` local hot across iterations. False means the
        lowering will emit :func:`aie.cascade_flow`.
        """
        return self._same_tile

    @property
    def producer_tile(self) -> Tile:
        return self._prod_tile

    @property
    def consumer_tile(self) -> Tile:
        return self._cons_tile

    def prod(self) -> "AccumFifoHandle":
        """Return the producer handle. AccumFifo is point-to-point: a
        second call returns the same handle rather than constructing a
        new one (mirrors :meth:`ObjectFifo.prod`).
        """
        if self._prod is None:
            self._prod = AccumFifoHandle(self, is_prod=True)
        return self._prod

    def cons(self) -> "AccumFifoHandle":
        """Return the consumer handle. AccumFifo is point-to-point:
        unlike :class:`ObjectFifo` (which supports multiple consumers
        for broadcast), an accumulator channel has exactly one consumer.
        AM020 Ch. 4 p. 67's cascade-stream description is unambiguously
        a chain (one source, one sink per hop), and the BM-to-BM
        register move is by definition point-to-point.
        """
        if self._cons is None:
            self._cons = AccumFifoHandle(self, is_prod=False)
        return self._cons

    def __str__(self) -> str:
        kind = "intra-tile" if self._same_tile else "cascade"
        return (
            f"AccumFifo(name={self.name!r}, dtype={self._dtype!r}, "
            f"lanes={self._lanes}, kind={kind}, "
            f"prod_tile=(col={self._prod_tile.col}, row={self._prod_tile.row}), "
            f"cons_tile=(col={self._cons_tile.col}, row={self._cons_tile.row}))"
        )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Lower the AccumFifo to MLIR.

        - Intra-tile (``producer.tile == consumer.tile``): no op emitted.
          The accumulator-register continuity is the C++ kernel's job
          (its ``aie::accum<T,N>`` local survives the worker's
          ``while(true)`` body); IRON has nothing to lower here.

        - Inter-tile: emit :func:`aie.cascade_flow` between the two
          tile ops. The placement pass converts this into a pair of
          :func:`aie.configure_cascade` ops at codegen time.

        Re-entrancy is guarded the same way :class:`ObjectFifo` guards
        it (set ``self._resolving = True`` before recursing into
        endpoint resolution; idempotent if called twice).
        """
        if self._resolving:
            return
        self._resolving = True

        if self._same_tile:
            self._op = None
            return

        # Inter-tile: emit aie.cascade_flow between the two tiles.
        # The tiles must already have their .op set by the placement
        # pass; if not, this raises NotResolvedError downstream.
        prod_op = self._prod_tile.op
        cons_op = self._cons_tile.op
        self._op = cascade_flow(prod_op, cons_op, loc=loc, ip=ip)

    @property
    def op(self):
        """The lowered :func:`aie.cascade_flow` op, or ``None`` if intra-tile.

        Intra-tile AccumFifos legitimately have no MLIR op -- the
        BM-to-BM register move is the C++ kernel's responsibility, not
        the dialect's. Callers that want to differentiate should branch
        on :attr:`is_intra_tile` instead of asserting ``op is not None``.
        """
        if not self._resolving:
            raise NotResolvedError()
        return self._op


class AccumFifoHandle(ObjectFifoHandle):
    """Producer or consumer handle to an :class:`AccumFifo`.

    Subclasses :class:`ObjectFifoHandle` so any code path that
    type-checks against ``ObjectFifoHandle`` (placer, validator, debug
    dumps) sees a uniform "fifo handle" surface. The
    ``fifo_handle_registry`` dispatch is the load-bearing
    ``Worker.fn_args`` path; the inheritance is kept for surface
    compatibility with code that hasn't migrated to the registry.

    The class deliberately overrides ``acquire`` / ``release`` to be
    no-ops in the intra-tile case (BM-to-BM register move has no
    DMA-channel synchronization to model) and to use the
    :class:`AccumFifo`'s point-to-point invariant in the cascade case
    (a cascade transfer takes exactly one accumulator per cycle; there
    is no notion of a "circular buffer with depth N" here).
    """

    def __init__(self, accum_fifo: AccumFifo, is_prod: bool):
        # Bypass ObjectFifoHandle.__init__ -- it requires an `of` with
        # ObjectFifo semantics (depth, dims_from_stream_per_cons, etc.)
        # that AccumFifo does not have. Set the fields ObjectFifoHandle
        # exposes as properties directly so isinstance-based downstream
        # code keeps working.
        self._port = (
            ObjectFifoPort.Produce if is_prod else ObjectFifoPort.Consume
        )
        self._is_prod = is_prod
        self._accum_fifo = accum_fifo
        # AccumFifo doesn't use ObjectFifo depth -- a cascade transfer
        # is one-accumulator-per-cycle, no circular buffer. Expose
        # depth=1 so any caller that defensively reads .depth gets a
        # sensible answer instead of None.
        self._depth = 1
        self._endpoint: ObjectFifoEndpoint | None = None
        self._dims_from_stream = None
        # Bind endpoint to the AccumFifo's pre-placed tile so
        # placement-aware passes see the wire-up.
        self._tile: Tile = (
            accum_fifo.producer_tile if is_prod else accum_fifo.consumer_tile
        )

    @property
    def name(self) -> str:
        return self._accum_fifo.name

    @property
    def handle_type(self) -> str:
        return "prod" if self._is_prod else "cons"

    @property
    def accum_fifo(self) -> AccumFifo:
        """The underlying :class:`AccumFifo`."""
        return self._accum_fifo

    @property
    def dtype(self) -> str:  # type: ignore[override]
        """Accumulator dtype tag (overrides ObjectFifoHandle's numpy dtype)."""
        return self._accum_fifo.dtype

    @property
    def lanes(self) -> int:
        return self._accum_fifo.lanes

    @property
    def shape(self) -> Sequence[int]:  # type: ignore[override]
        """Per-transfer shape: a flat (lanes,) accumulator slice."""
        return (self._accum_fifo.lanes,)

    @property
    def obj_type(self):  # type: ignore[override]
        """AccumFifo has no memref obj_type -- the accumulator register
        is not a memref word. Callers that need a numpy-typed view of a
        cascade transfer should use ``np.zeros((handle.lanes,), np.float32)``
        for ``accfloat`` (the FP32 case).
        """
        if self._accum_fifo.dtype == "accfloat":
            return np.ndarray[(self._accum_fifo.lanes,), np.dtype(np.float32)]
        if self._accum_fifo.dtype == "acc32":
            return np.ndarray[(self._accum_fifo.lanes,), np.dtype(np.int32)]
        if self._accum_fifo.dtype == "acc64":
            return np.ndarray[(self._accum_fifo.lanes,), np.dtype(np.int64)]
        # _validate_dtype already rejected anything else; defensive only.
        raise ValueError(f"AccumFifo: unexpected dtype {self._accum_fifo.dtype!r}")

    def acquire(self, num_elem: int = 1):
        """Acquire ``num_elem`` accumulator transfers.

        For an :class:`AccumFifo` the synchronization model is:

        - Intra-tile: no-op. BM-to-BM register move is implicit in the
          C++ kernel keeping its ``aie::accum`` local across iterations.
        - Inter-tile: implicit per-cycle handshake on the cascade wire.
          The C++ kernel's ``get_scd`` / ``put_mcd`` intrinsic IS the
          synchronization. There is no IRON-level lock to take.

        Returns ``None`` (no buffer to index into -- the accumulator
        lives in the kernel's register file). The signature mirrors
        :meth:`ObjectFifoHandle.acquire` so user code parameterized
        over both fifo kinds doesn't need a branch.
        """
        if num_elem != 1:
            raise ValueError(
                f"AccumFifo handle: acquire() takes exactly 1 cascade transfer "
                f"per call (1 accumulator = 1 cycle on the cascade wire); "
                f"got num_elem={num_elem}"
            )
        return None

    def release(self, num_elem: int = 1) -> None:
        """Release ``num_elem`` accumulator transfers.

        Symmetric to :meth:`acquire`: a no-op at the IRON level. The
        cascade wire's per-cycle handshake (and, for the intra-tile
        case, the kernel's register-file aliasing) does the actual
        synchronization.
        """
        if num_elem != 1:
            raise ValueError(
                f"AccumFifo handle: release() takes exactly 1 cascade transfer "
                f"per call; got num_elem={num_elem}"
            )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Forward to the underlying :class:`AccumFifo` (idempotent)."""
        self._accum_fifo.resolve(loc=loc, ip=ip)

    def all_of_endpoints(self) -> list[ObjectFifoEndpoint]:  # type: ignore[override]
        """All endpoints (producer + consumer) of the AccumFifo.

        Returns endpoint-typed objects exposing a ``.tile`` attribute,
        matching :meth:`ObjectFifoHandle.all_of_endpoints`'s contract.
        ``iron/program.py`` walks every fifo and does
        ``[e.tile for e in fifo.all_of_endpoints()]`` to populate the
        device's tile-resolution set.

        :class:`AccumFifoHandle.__init__` deliberately bypasses
        :class:`ObjectFifoHandle.__init__` (no memref-typed buffer; no
        ObjectFifo depth/dims to wire up), so the inherited
        ``all_of_endpoints`` -- which walks
        ``self._object_fifo._get_endpoint(...)`` -- would raise
        ``AttributeError: 'AccumFifoHandle' object has no attribute
        '_object_fifo'``. Override returns endpoints constructed
        directly from this handle's tile pair.

        For each side we prefer the live endpoint recorded on the
        corresponding handle by the registry-driven ``Worker.fn_args``
        dispatch (the :class:`Worker` instance, since :class:`Worker`
        subclasses :class:`ObjectFifoEndpoint`). When a handle has not
        yet been constructed we synthesize a bare
        :class:`ObjectFifoEndpoint` wrapping the AccumFifo's pre-placed
        :class:`Tile` so callers still get a uniform ``.tile`` view.
        """
        af = self._accum_fifo
        prod_handle = af._prod
        cons_handle = af._cons
        prod_ep = (
            prod_handle.endpoint
            if (prod_handle is not None and prod_handle.endpoint is not None)
            else ObjectFifoEndpoint(af.producer_tile)
        )
        cons_ep = (
            cons_handle.endpoint
            if (cons_handle is not None and cons_handle.endpoint is not None)
            else ObjectFifoEndpoint(af.consumer_tile)
        )
        return [prod_ep, cons_ep]

    def __str__(self) -> str:
        return (
            f"AccumFifoHandle({self.handle_type}, dtype={self.dtype!r}, "
            f"lanes={self.lanes}, of={self._accum_fifo})"
        )
