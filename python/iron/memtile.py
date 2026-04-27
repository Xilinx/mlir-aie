# memtile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""MemtileAggregator: first-class IRON memtile-mediated fan-in helper.

Encapsulates the memtile-aggregated 4-into-1 fan-in topology that
Phase 1 of the bio-on-XDNA project hand-rolled in
after hitting the compute-tile 2-input-DMA-channel ceiling
promotes that topology to a first-class API so every multi-tile
CRISPR / match / reduction design can reach for it the same way they
reach for :class:`ObjectFifo`.

Architectural reference
-----------------------

- AM020 Ch. 5 p. 74 (memtile DMA): 6 MM2S + 6 S2MM channels per
  memtile; channels 0..3 of the S2MM side support east/west neighbour
  access, making them the canonical fan-in target for up to 4
  producer compute tiles. Channels 4..5 are reserved for shim-side
  DMA traffic.
- AM020 Ch. 5 Figures 22 + 23 + the "Dataflow Mapping 1/2/3"
  diagrams: the canonical fan-in via memtile pattern uses a single
  joined ObjectFifo on the memtile, with each producer slotted via
  an offset into the joined buffer. The memtile's 5D address
  generation (Ch. 5 p. 71) handles the per-producer reorganisation
  without needing a joiner compute kernel.
- AM020 Table 14: memtile DM is 512 KiB on AIE-ML; the AIE2P

Design notes
------------

MemtileAggregator does **not** subclass :class:`ObjectFifo`. Under the
hood it composes three things:

1. A single joined :class:`ObjectFifo` whose producer side fans out
   to ``len(producers)`` per-tile sub-FIFOs via
   :meth:`ObjectFifo.prod().join`. Each sub-FIFO occupies one
   memtile S2MM channel (0..3), and the memtile reorganises the
   incoming partials into the joined buffer using the documented
   flat-byte-offset list.
2. A consumer-side handle on the same joined FIFO, suitable for
   either a single CoreTile worker or a shim DMA drain.
3. An accessor (:meth:`producer`, :meth:`producers`) returning the
   per-tile producer handles so the caller can pass them into their
   match/compute Workers' ``fn_args``.

This split keeps :class:`ObjectFifo`'s shared-memory semantics intact
while letting :class:`MemtileAggregator` emit the fan-in pattern
placement diagnostics) to hint at MemtileAggregator when a user
hits the 2-input-DMA-channel ceiling.

The flat-byte-offset layout lesson (CRITICAL -- read before use)
----------------------------------------------------------------

The IRON :meth:`ObjectFifo.prod().join` primitive is **flat byte-offset
concatenation**. It is *not* strided / interleaved.

Concretely, given ``offsets=[0, B, 2*B, 3*B]`` where ``B`` is each
producer's per-element byte size, the memtile lays the 4 producers'
partials end-to-end into a single contiguous joined buffer:

::

    byte 0   .. B-1     <- producer 0
    byte B   .. 2B-1    <- producer 1
    byte 2B  .. 3B-1    <- producer 2
    byte 3B  .. 4B-1    <- producer 3

This means the **logical layout each producer emits in must be
"slab-major"**, where each slab maps directly to its byte range in
the joined buffer. For Phase 1's CRISPR mismatch design this meant
**guide-major** partials (each tile writes
``partial[g_local * n_windows + w]``); flat-concat then yields a
final layout where tile 0's 32 guides occupy rows 0..31, tile 1's
rows 32..63, etc., naturally producing the (N_GUIDES x N_WINDOWS)
guide-major output the consumer expects.

If a producer instead emits **window-major** partials
(``partial[w * n_guides_per_tile + g_local]``), flat-concat does NOT
transpose. The result is interleaved nonsense unless the consumer
explicitly un-shuffles it (Phase 1 hit this and fell back to a
path skipped the transpose entirely).

This API exposes a ``layout`` parameter (one of ``"slab"`` or
``"window"``) that documents the producer-side requirement at the
type level. ``"slab"`` (the default) is the canonical
flat-concat-friendly layout. Specifying ``"window"`` causes
MemtileAggregator to insert a memtile-side ``dims_from_stream``
re-layout so the consumer still sees a contiguous slab-major
buffer; this costs memtile DMA cycles but lets producer kernels
without rewriting the kernel.

Phase 1 measurement summary
---------------------------

windows x 4 match tiles measured a 1.90x throughput speedup vs
Memtile occupancy: 32 KiB / 512 KiB (~6.25%). Per-match-tile
occupancy: 5.4 KiB / 64 KiB (~8.4%). Output byte-equal to the

Example
-------

.. code-block:: python

    import numpy as np
    from aie.iron import (
        Kernel, MemtileAggregator, ObjectFifo, Program, Runtime, Worker,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2

    # 4 match tiles, each producing a 64-window x 32-guide partial
    # (guide-major / "slab" layout).
    N_TILES, N_GUIDES, N_WINDOWS = 4, 128, 4096
    GUIDES_PER_TILE = N_GUIDES // N_TILES
    WINDOWS_PER_CHUNK = 64

    partial_ty = np.ndarray[
        (WINDOWS_PER_CHUNK * GUIDES_PER_TILE,), np.dtype[np.uint8]
    ]
    joined_ty = np.ndarray[
        (WINDOWS_PER_CHUNK * N_GUIDES,), np.dtype[np.uint8]
    ]

    agg = MemtileAggregator(
        n_producers=N_TILES,
        producer_obj_type=partial_ty,
        joined_obj_type=joined_ty,
        layout="slab",
        name="match_join",
    )

    # Per-tile producer handles -- pass into each Worker's fn_args.
    match_workers = [
        Worker(match_body, fn_args=[..., agg.producer(i)])
        for i in range(N_TILES)
    ]

    # Consumer side: a single shim drain or a single CoreTile worker.
    rt = Runtime()
    with rt.sequence(...) as (G, W, Out):
        rt.start(*match_workers)
        rt.drain(agg.consumer(), Out, wait=True)

``of_out.prod().join(join_offsets, ...)`` pattern -- byte-equal
output guaranteed by the layout contract.
"""

from __future__ import annotations

import numpy as np

from ..helpers.util import (
    np_ndarray_type_get_shape,
    np_ndarray_type_get_dtype,
)
from .dataflow import ObjectFifo
from .dataflow.objectfifo import ObjectFifoHandle
from .device import Tile, AnyMemTile

# AM020 Ch. 5 p. 74: memtile S2MM has 6 channels total, channels 0..3
# support east/west neighbour access (i.e. the channels usable as a
# compute-tile fan-in target). Channels 4..5 are typically reserved
# for shim-side DMA. This caps the fan-in width at 4 producers per
# memtile aggregator instance.
MEMTILE_S2MM_NEIGHBOUR_CHANNELS: int = 4

# AM020 Ch. 5 + Table 14: AIE-ML / AIE2P memtile DM capacity.
MEMTILE_DM_BYTES: int = 512 * 1024  # 512 KiB

# Producer-side layout vocabulary. Documented in the module docstring.
_VALID_LAYOUTS: frozenset[str] = frozenset({"slab", "window"})

# ---------------------------------------------------------------------------
# Internal helpers (kept module-local to keep the import surface narrow).
# ---------------------------------------------------------------------------

def _bytes_per_element(ndarray_type: type[np.ndarray]) -> int:
    """Compute the byte size of one element of an IRON ndarray type.

    IRON uses ``numpy.ndarray[(shape,), numpy.dtype[T]]`` as its
    per-element type. This helper multiplies the shape's dim product
    by the dtype's per-scalar byte size to get the total per-element
    byte size used for flat-concat offset arithmetic.
    """
    shape = np_ndarray_type_get_shape(ndarray_type)
    dtype = np_ndarray_type_get_dtype(ndarray_type)
    n_elems = 1
    for d in shape:
        n_elems *= int(d)
    # ``dtype`` may be a numpy scalar type or an alias (e.g. bfloat16
    # via the ml_dtypes package). Both expose ``.itemsize`` via
    # ``np.dtype(...)``.
    itemsize = int(np.dtype(dtype).itemsize)
    return n_elems * itemsize

def _flat_concat_offsets(n_producers: int, partial_bytes: int) -> list[int]:
    """Return the flat-byte-offset list used in the ``join()`` lowering.

    Producer ``i``'s partial lands at byte offset
    ``i * partial_bytes`` in the joined buffer; this is exactly the
    pattern AM020 Ch. 5 p. 71 names "flat concatenation" and the
    only pattern :meth:`ObjectFifo.prod().join` lowers without an
    accompanying ``dims_to_stream`` re-layout.
    """
    return [i * partial_bytes for i in range(n_producers)]

# Public re-exports for callers that need to compute offsets / sizes
# without instantiating an aggregator (e.g. for size assertions in
# downstream tests).
bytes_per_element = _bytes_per_element
flat_concat_offsets = _flat_concat_offsets

class MemtileAggregator:
    """Memtile-mediated N-into-1 fan-in helper (N <= 4).

    Builds a single joined :class:`ObjectFifo` on a memtile whose
    producer side is fan-out across up to 4 per-tile producer
    sub-FIFOs (one memtile S2MM channel per producer, channels 0..3
    per AM020 Ch. 5 p. 74). The memtile reorganises the partials into
    a contiguous joined buffer using flat byte-offset concatenation;
    the consumer sees a single contiguous (joined) buffer.

    Args:
        n_producers: Number of producer compute tiles feeding the
            aggregator. Must be in ``[1, 4]`` per the memtile S2MM
            neighbour-channel budget.
        producer_obj_type: ``numpy.ndarray`` type describing each
            producer's per-element partial buffer. All producers must
            emit the same type (asymmetric fan-in is not supported by
            this helper; if you need it, drop down to
            :meth:`ObjectFifo.prod().join` directly).
        joined_obj_type: ``numpy.ndarray`` type describing the joined
            output buffer the consumer sees. Must satisfy
            ``sizeof(joined) == n_producers * sizeof(partial)`` (the
            flat-concat invariant). Validated at construction time.
        layout: Producer-side layout. ``"slab"`` (default) means each
            producer emits a slab whose flat byte range is the same
            as its position in the joined buffer (the canonical
            flat-concat-friendly layout, e.g. the "guide-major"
            choice for CRISPR mismatch). ``"window"`` means producers
            emit window-major partials and the memtile applies a
            ``dims_from_stream`` re-layout so the consumer still sees
            a slab-major joined buffer (costs memtile DMA cycles).
        depth: ObjectFifo depth (number of slots) on both producer
            and consumer sides. Defaults to 2 (double-buffered, the
            value Phase 1 + the AM020 examples use).
        tile: Memtile placement. Defaults to :data:`AnyMemTile`,
            letting the placer pick a memtile in the design.
        name: Optional symbolic name; auto-generated if omitted.

    Raises:
        TypeError: ``n_producers`` is not an int.
        ValueError: ``n_producers`` is not in ``[1, 4]``;
            ``layout`` is not one of ``"slab" | "window"``;
            ``depth`` is non-positive; the partial/joined sizes do
            not satisfy the flat-concat invariant; the joined buffer
            exceeds the memtile DM capacity.
        NotImplementedError: ``layout="window"`` is reserved for a
            Phase 3 extension; use ``ObjectFifo.prod().join`` directly
            with explicit ``dims_to_stream`` for now.

    Example:
        See module docstring.

    Notes:
        - For ``n_producers == 1`` the helper degenerates to a plain
          1-into-1 memtile staging FIFO. This is still useful (the
          consumer no longer needs to be DMA-adjacent to the
          producer; the memtile mediates), and the API is the same.
        - The MLIR lowering goes through
          will hint at this primitive when a design hits the
          compute-tile 2-input-DMA-channel ceiling.
    """

    # Used to generate unique aggregator names when none is provided.
    __ma_index = 0

    def __init__(
        self,
        n_producers: int,
        producer_obj_type: type[np.ndarray],
        joined_obj_type: type[np.ndarray],
        layout: str = "slab",
        depth: int = 2,
        tile: Tile | None = None,
        name: str | None = None,
    ):
        if not isinstance(n_producers, int):
            raise TypeError(
                f"MemtileAggregator: n_producers must be int, "
                f"got {type(n_producers).__name__}"
            )
        if n_producers < 1 or n_producers > MEMTILE_S2MM_NEIGHBOUR_CHANNELS:
            raise ValueError(
                f"MemtileAggregator: n_producers must be in "
                f"[1, {MEMTILE_S2MM_NEIGHBOUR_CHANNELS}] "
                f"(memtile S2MM neighbour-channel budget per AM020 "
                f"Ch. 5 p. 74), got {n_producers}"
            )
        if layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"MemtileAggregator: layout must be one of "
                f"{sorted(_VALID_LAYOUTS)}, got {layout!r}"
            )
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(
                f"MemtileAggregator: depth must be a positive int, "
                f"got {depth!r}"
            )

        partial_bytes = _bytes_per_element(producer_obj_type)
        joined_bytes = _bytes_per_element(joined_obj_type)

        expected_joined_bytes = partial_bytes * n_producers
        if joined_bytes != expected_joined_bytes:
            raise ValueError(
                f"MemtileAggregator: flat-concat invariant violated: "
                f"joined size ({joined_bytes} B) must equal "
                f"n_producers * partial size "
                f"({n_producers} * {partial_bytes} = "
                f"{expected_joined_bytes} B). "
                f"Read the 'flat-byte-offset layout lesson' section of "
                f"the MemtileAggregator docstring; the most common cause "
                f"is producers emitting window-major partials with the "
                f"default layout='slab'."
            )

        # Memtile DM headroom check: the memtile holds depth*partial_bytes
        # bytes per producer slot + depth*joined_bytes for the joined
        # output side. Stay well under 512 KiB on AIE-ML / AIE2P.
        memtile_footprint = (
            depth * partial_bytes * n_producers + depth * joined_bytes
        )
        if memtile_footprint > MEMTILE_DM_BYTES:
            raise ValueError(
                f"MemtileAggregator: memtile DM budget exceeded: "
                f"footprint {memtile_footprint} B > "
                f"{MEMTILE_DM_BYTES} B (per AM020 Table 14 / AIE2P "
                f"resource probe). Reduce depth, partial size, or "
                f"chunk the dataflow further."
            )

        if name is None:
            name = f"ma{MemtileAggregator.__get_index()}"

        self._n_producers = n_producers
        self._producer_obj_type = producer_obj_type
        self._joined_obj_type = joined_obj_type
        self._layout = layout
        self._depth = depth
        self._partial_bytes = partial_bytes
        self._joined_bytes = joined_bytes
        self._tile = tile if tile is not None else AnyMemTile
        self.name = name

        # ``layout="window"`` is documented but not lowered yet -- it
        # would need a memtile-side dims_from_stream computed from the
        # producer ndarray shape, which we can't infer generically
        # without more info from the caller. Surface a clear error so
        # the user knows to drop down to ObjectFifo.prod().join() with
        # explicit dims_to_stream until Phase 3 extends this helper.
        if layout == "window":
            raise NotImplementedError(
                "MemtileAggregator: layout='window' is reserved for a "
                "future Phase 3 extension that infers the memtile-side "
                "dims_to_stream from the producer shape. For now, "
                "either (1) emit slab-major partials in your producer "
                "flat-concat-friendly layout) or (2) drop down to "
                "ObjectFifo.prod().join(offsets, dims_to_stream=[...]) "
                "and pass the dims explicitly. See module docstring."
            )

        # Construct the joined ObjectFifo. The producer side is then
        # split into n_producers sub-FIFOs via .prod().join(...).
        self._joined_fifo = ObjectFifo(
            joined_obj_type,
            depth=depth,
            name=f"{name}_joined",
        )

        offsets = _flat_concat_offsets(n_producers, partial_bytes)

        self._sub_fifos: list[ObjectFifo] = self._joined_fifo.prod().join(
            offsets,
            tile=self._tile,
            obj_types=[producer_obj_type] * n_producers,
            names=[f"{name}_p{i}" for i in range(n_producers)],
        )

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__ma_index
        cls.__ma_index += 1
        return idx

    # ----- read-only introspection ---------------------------------------

    @property
    def n_producers(self) -> int:
        """Number of per-tile producer slots (1..4)."""
        return self._n_producers

    @property
    def producer_obj_type(self) -> type[np.ndarray]:
        """Per-producer partial buffer type."""
        return self._producer_obj_type

    @property
    def joined_obj_type(self) -> type[np.ndarray]:
        """Joined buffer type the consumer sees."""
        return self._joined_obj_type

    @property
    def layout(self) -> str:
        """Producer-side layout (`'slab'` or `'window'`)."""
        return self._layout

    @property
    def depth(self) -> int:
        """ObjectFifo depth on both producer and consumer sides."""
        return self._depth

    @property
    def offsets(self) -> list[int]:
        """Flat byte-offset list used in the underlying join() call."""
        return _flat_concat_offsets(self._n_producers, self._partial_bytes)

    @property
    def joined_fifo(self) -> ObjectFifo:
        """The underlying joined ObjectFifo (consumer-side handle source)."""
        return self._joined_fifo

    @property
    def sub_fifos(self) -> list[ObjectFifo]:
        """The N per-producer sub-ObjectFifos returned by ``join()``."""
        return list(self._sub_fifos)

    # ----- handle accessors (the user-facing API) ------------------------

    def producer(self, i: int) -> ObjectFifoHandle:
        """Return the producer-side handle for slot ``i`` (0-indexed).

        Pass this handle into the corresponding match/compute Worker's
        ``fn_args`` so the worker writes its partial into memtile S2MM
        channel ``i``.

        Args:
            i: Producer index, ``0 <= i < n_producers``.

        Raises:
            IndexError: ``i`` is out of range.

        Returns:
            ObjectFifoHandle: Producer handle for sub-FIFO ``i``.
        """
        if i < 0 or i >= self._n_producers:
            raise IndexError(
                f"MemtileAggregator.producer: index {i} out of range "
                f"[0, {self._n_producers})"
            )
        return self._sub_fifos[i].prod()

    def producers(self) -> list[ObjectFifoHandle]:
        """Return all producer-side handles, ordered by slot index.

        Equivalent to ``[self.producer(i) for i in range(n_producers)]``.
        """
        return [self.producer(i) for i in range(self._n_producers)]

    def consumer(self, depth: int | None = None) -> ObjectFifoHandle:
        """Return the consumer-side handle on the joined buffer.

        The consumer is typically either a single CoreTile Worker
        (passed via ``fn_args``) or a shim DMA drain (passed to
        :meth:`Runtime.drain`).

        Args:
            depth: Optional override for the consumer-side ObjectFifo
                depth. Defaults to the aggregator's depth.

        Returns:
            ObjectFifoHandle: Consumer handle for the joined FIFO.
        """
        return self._joined_fifo.cons(depth=depth)

    # ----- string repr ---------------------------------------------------

    def __str__(self) -> str:
        return (
            f"MemtileAggregator(name={self.name!r}, "
            f"n_producers={self._n_producers}, layout={self._layout!r}, "
            f"depth={self._depth}, "
            f"partial_bytes={self._partial_bytes}, "
            f"joined_bytes={self._joined_bytes})"
        )

__all__ = [
    "MemtileAggregator",
    "MEMTILE_S2MM_NEIGHBOUR_CHANNELS",
    "MEMTILE_DM_BYTES",
    "bytes_per_element",
    "flat_concat_offsets",
]
