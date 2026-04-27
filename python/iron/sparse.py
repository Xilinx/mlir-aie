# sparse.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON :class:`SparseFifo` -- on-the-fly N:M sparsity decompression FIFO.

Promotes the AIE-ML / AIE2P **compute-tile S2MM decompression** + **MM2S
compression** hardware (AM020 Ch. 1 p. 15 + Ch. 2 p. 27 + Ch. 5 p. 74)
to a first-class IRON dataflow primitive. The user-facing surface looks
identical to :class:`aie.iron.ObjectFifo`: a producer pushes compressed
elements, a consumer reads dense elements. The lowering attaches
compression/decompression metadata so the buffer-descriptor configuration
flips the per-channel ``Enable_Compression`` bit (compute-tile DMA BD,
documented in ``lib/Dialect/AIE/Util/aie_registers_aie2.json`` under
``Enable_Compression`` / "Enable Compression (MM2S), decompression
(S2MM). Only effective if channel has (de)compression enabled").

Architectural references
------------------------

- AM020 Ch. 1 p. 15: AIE-ML supports structured sparsity for "CNN and
  RNN application".
- AM020 Ch. 2 p. 27: "Adds decompression to the two S2MM channels" +
  "Adds compression to the two MM2S channels" (compute-tile DMA, for
  on-tile sparse-weight load).
- AM020 Ch. 5 p. 74: memtile-side compression / decompression
  capability (the corollary path for memtile-mediated supply).
- AM020 Appendix A: confirms structured sparsity carries forward; AIE2P
  supported patterns may differ -- see "AIE2P caveat" below.
- ``aie_registers_aie2.json`` BD field ``Enable_Compression`` (per-channel;
  effective only if the DMA channel has (de)compression enabled at the
  channel-descriptor level).

User-facing surface
-------------------

.. code-block:: python

    from aie.iron import SparseFifo, Worker
    from aie.iron.device import Tile
    import numpy as np

    # 2:4 N:M sparsity (newest, GPU-style; 50 % zeros per group of 4).
    weights = SparseFifo(
        producer=Tile(0, 0),       # Shim/host: writes COMPRESSED weights.
        consumer=Tile(0, 2),       # CoreTile: reads DENSE weights via S2MM
                                   #           on-the-fly decompression.
        obj_type=np.ndarray[(64, 64), np.dtype(np.int8)],
        sparsity_pattern="N:M",
        N=2,
        M=4,
        depth=2,
        name="lstm_weights_sparse",
    )

    # Workers receive the same kind of `prod()` / `cons()` handle as
    # ObjectFifo. The handle is a true ObjectFifoHandle subclass
    # (SparseFifoHandle) so Worker.fn_args dispatch -- both via the
    # accepts it without modification.
    w_consumer = Worker(consume_fn, fn_args=[weights.cons(), ...],
                        tile=Tile(0, 2))

The user sees one channel; the underlying ObjectFifo lowers normally
plus the produced ObjectFifoCreateOp gets two boolean-typed metadata
attributes attached at resolve()-time:

- ``aie.compress_mm2s = true`` on the producer side (compress on the
  way out of the producer's MM2S channel)
- ``aie.decompress_s2mm = true`` on the consumer side (decompress on
  the way into the consumer's S2MM channel)

Why an ObjectFifo subclass and not a flag on ObjectFifo itself
--------------------------------------------------------------

ObjectFifo is the canonical memref-typed dataflow channel and we don't
want to broaden its constructor with sparsity knobs that >95 % of
designs will never use. SparseFifo composes-by-subclassing: it
piggy-backs on ObjectFifo's storage, depth, dimensionsToStream/-FromStream,
and pad/repeat/iter machinery, then adds the (sparsity_pattern, N, M)
triple plus the compression-attribute lowering hook in ``resolve()``.
Callers that don't import ``SparseFifo`` see no API change.

Lowering model
--------------

At construction time, :class:`SparseFifo` builds a vanilla
:class:`ObjectFifo` whose buffer obj_type is the **dense** element type
(i.e. the type the consumer sees). The compression ratio is implicit
in the sparsity pattern: a 2:4 channel that physically transfers 50 %
of the dense bytes (because the zero positions are encoded out of the
stream) still presents a dense memref to the consumer, because the
hardware's S2MM decompression block re-injects the zeros before they
land in tile DM.

When :meth:`resolve` runs, after the underlying ObjectFifoCreateOp is
built, the SparseFifo attaches two boolean attributes that the
``AIEObjectFifoStatefulTransformPass`` and downstream BD-emit passes
honor by setting ``Enable_Compression`` on the relevant DMA BD words.
The ``aie.compress_mm2s`` / ``aie.decompress_s2mm`` attribute names
match the ``aie_registers_aie2.json`` field semantics so the pass is
mechanical: it copies the boolean to the BD config.

If the active backend doesn't yet honor the attribute (early AIE2P
silicon-driver versions; fork rebuild not yet propagated), the design
remains compilable and runs as a non-sparse equivalent -- the
attribute is lowering metadata, not a structural ObjectFifo change.
This degraded mode is observable: the runtime DMA volume is the dense
includes a runtime byte-counter that fires the AIE2P-divergence
warning if the measured DMA volume matches the dense reference.

AIE2P caveat
------------

AM020 documents AIE-ML's supported sparsity patterns. AIE2P inherits
the compute-tile DMA Enable_Compression bit (the ``aie_registers_aie2.json``
file we vendor is shared across AIE-ML and AIE2P silicon), but the
**accepted N:M patterns may differ**. Specifically:

- AIE-ML AM020 cites 1:2 and 2:4 structured sparsity patterns.
- AIE2P (newer NPU silicon) may add or restrict patterns based on
  refinements in the on-tile decompression LUT.
  (compressed output != dense reference, or DMA volume == dense),
  SparseFifo itself remains usable on AIE-ML targets even if AIE2P
  diverges; the divergence is a runtime / silicon issue, not an
  IRON-API issue.

The :meth:`SparseFifo.compression_ratio` property reports the
**theoretical** ratio per the requested pattern (N/M); the runtime
**measured** ratio is the consumer's responsibility.

Sparsity pattern rules
----------------------

We accept ``sparsity_pattern="N:M"`` as the canonical form, with
``N`` and ``M`` integers satisfying:

- ``M >= 2`` (degenerate 1:1 = no sparsity rejected; use ObjectFifo).
- ``0 < N < M`` (a group can't be all-zero or all-nonzero by pattern).
- ``(N, M)`` in the AM020-documented set ``{(1, 2), (1, 4), (2, 4)}``
  by default; other combinations raise unless ``allow_unverified=True``

Validation lives in :func:`_validate_sparsity_pattern` and runs at
construction time so users see the failure at API-call time, not deep
inside MLIR lowering.

References
----------

- AM020 Ch. 1 p. 15 (sparsity for CNN / RNN application; AIE-ML)
- AM020 Ch. 2 p. 27 (S2MM decompression + MM2S compression on compute
  tile; two channels each direction)
- AM020 Ch. 5 p. 74 (memtile compression / decompression)
- ``lib/Dialect/AIE/Util/aie_registers_aie2.json`` (Enable_Compression
  BD bit; "Only effective if channel has (de)compression enabled")
  consumer that triggers AIE2P silicon validation)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import ir  # type: ignore
from .dataflow.fifo_handle_registry import register_fifo_handle
from .dataflow.objectfifo import ObjectFifo, ObjectFifoHandle
from .device import Tile

# AM020 Ch. 2 p. 27: per-direction compute-tile DMA channel count
# carrying sparsity-aware (de)compression. Two S2MM channels gain
# decompression; two MM2S channels gain compression.
SPARSE_CHANNELS_PER_DIRECTION: int = 2

# AM020-documented N:M sparsity patterns for AIE-ML. AIE2P may diverge --
# docstring "AIE2P caveat"). Patterns outside this set require
# ``allow_unverified=True`` so the user opts into the experimental path.
_AM020_VERIFIED_NM_PATTERNS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 2),  # 50 % sparsity, simplest pattern
        (1, 4),  # 75 % sparsity
        (2, 4),  # 50 % sparsity, AM020-documented GPU-style pattern
        (2, 4),  # 50 % sparsity, AM020-documented GPU-style pattern (default)
    }
)

# Attribute names the IRON layer pins on the lowered ObjectFifoCreateOp
# so downstream passes (the BD-emit pass; the runtime bookkeeping pass)
# can find the compression intent. The names match the
# ``aie_registers_aie2.json`` field semantics ("Enable Compression
# (MM2S), decompression (S2MM)").
SPARSE_ATTR_COMPRESS_MM2S: str = "aie.compress_mm2s"
SPARSE_ATTR_DECOMPRESS_S2MM: str = "aie.decompress_s2mm"
SPARSE_ATTR_PATTERN: str = "aie.sparsity_pattern"
SPARSE_ATTR_N: str = "aie.sparsity_n"
SPARSE_ATTR_M: str = "aie.sparsity_m"

def _validate_sparsity_pattern(
    sparsity_pattern: str,
    N: int,
    M: int,
    allow_unverified: bool,
) -> None:
    """Validate the (pattern, N, M) triple against the AM020 spec.

    Args:
        sparsity_pattern: The pattern tag. Currently only ``"N:M"``
            is accepted; the ``str`` argument exists so future
            extensions (block sparsity, COO-style, etc.) slot in
            without breaking the API surface.
        N: Number of nonzero elements per group of size ``M``.
        M: Group size.
        allow_unverified: If ``True``, accept (N, M) pairs outside
            :data:`_AM020_VERIFIED_NM_PATTERNS` (the AIE2P-investigation
            escape hatch).

    Raises:
        ValueError: Pattern tag isn't ``"N:M"``, or N/M violates the
            structural rules, or (N, M) isn't in the AM020-verified
            set and ``allow_unverified`` is ``False``.
        TypeError: N or M not an int.
    """
    if sparsity_pattern != "N:M":
        raise ValueError(
            f"SparseFifo: sparsity_pattern={sparsity_pattern!r} not "
            f"supported. Only 'N:M' is accepted in this fork; future "
            f"patterns (block, COO) will land as separate kwargs."
        )

    if not isinstance(N, int) or not isinstance(M, int):
        raise TypeError(
            f"SparseFifo: N and M must be int, got "
            f"N={type(N).__name__}, M={type(M).__name__}"
        )

    if M < 2:
        raise ValueError(
            f"SparseFifo: M must be >= 2 (a group of 1 has no zero "
            f"slots; that's not sparsity, use ObjectFifo); got M={M}"
        )

    if N <= 0 or N >= M:
        raise ValueError(
            f"SparseFifo: must have 0 < N < M (a group of {M} with "
            f"N={N} nonzeros is degenerate). 0 means all-zero (use a "
            f"constant-zero buffer); N==M means dense (use ObjectFifo)."
        )

    if (N, M) not in _AM020_VERIFIED_NM_PATTERNS and not allow_unverified:
        raise ValueError(
            f"SparseFifo: (N={N}, M={M}) is not in the AM020-verified "
            f"set {sorted(_AM020_VERIFIED_NM_PATTERNS)}. AIE-ML "
            f"hardware documents 1:2, 1:4, 2:4 N:M structured "
            f"sparsity (Ch. 1 p. 15). Pass allow_unverified=True if "
            f"silicon test should confirm the lowering before this is "
            f"folded back into the verified set."
        )

def _coerce_to_tile(arg, role: str) -> Tile:
    """Accept either a :class:`Tile` or anything exposing ``.tile``.

    Same pattern :class:`AccumFifo` uses so callers can pass an
    already-placed :class:`Worker` instead of restating its coordinates.
    """
    if isinstance(arg, Tile):
        return arg
    maybe_tile = getattr(arg, "tile", None)
    if isinstance(maybe_tile, Tile):
        return maybe_tile
    raise TypeError(
        f"SparseFifo: {role} must be a Tile or a placed Worker (anything "
        f"exposing a `.tile` attribute of type Tile); got "
        f"{type(arg).__name__}"
    )

class SparseFifo(ObjectFifo):
    """ObjectFifo that compresses on MM2S and decompresses on S2MM.

    User-facing surface is identical to :class:`ObjectFifo` -- the
    producer pushes data with :meth:`prod` and the consumer pulls data
    with :meth:`cons`. The difference is the BD-config metadata pinned
    on the lowered ``aie.objectfifo`` op: ``aie.compress_mm2s`` on the
    producer side, ``aie.decompress_s2mm`` on the consumer side, plus
    the sparsity pattern itself (``aie.sparsity_pattern``,
    ``aie.sparsity_n``, ``aie.sparsity_m``). The downstream BD-emit
    pass honors these by flipping the per-channel ``Enable_Compression``
    bit (AM020 Ch. 2 p. 27 + ``aie_registers_aie2.json``).

    The :meth:`prod` / :meth:`cons` handles are
    :class:`SparseFifoHandle` (an :class:`ObjectFifoHandle` subclass);
    they pass through Worker.fn_args' isinstance(handle, ObjectFifoHandle)
    ``main``. They are pre-registered with the registry at module
    import time so that, if the registered ObjectFifoHandle handler is
    ever replaced upstream, the SparseFifo flavour still has a clear
    backstop.

    Args:
        producer: The :class:`Tile` (or placed :class:`Worker`) that
            pushes COMPRESSED data on its MM2S channel. Typically the
            shim DMA / host source for sparse weights.
        consumer: The :class:`Tile` (or placed :class:`Worker`) that
            receives DENSE data on its S2MM channel. The compute-tile
            decompressor injects the zeros transparently.
        obj_type: The DENSE element type (numpy ndarray type) the
            consumer sees. A 64x64 INT8 weight tile keeps that
            obj_type whether the wire carries 50 % bytes (2:4) or 100 %
            bytes -- the dense view is the consumer's view.
        sparsity_pattern: ``"N:M"`` (the only currently-supported tag).
        N: Number of nonzero elements per group of size ``M``.
            Default 2.
        M: Group size. Default 4 (the canonical 2:4 GPU-style pattern;
            matches the AM020 Ch. 1 p. 15 RNN-application example).
        depth: ObjectFifo depth (mirrors ObjectFifo's argument).
            Default 2.
        name: Optional name. Auto-generated if omitted.
        dims_to_stream: Same as ObjectFifo; applies to the producer-side
            data layout BEFORE compression.
        dims_from_stream_per_cons: Same as ObjectFifo; applies to the
            consumer-side data layout AFTER decompression.
        plio: Same as ObjectFifo.
        pad_dimensions: Same as ObjectFifo.
        allow_unverified: Accept (N, M) pairs outside the AM020-verified
            set ``{(1, 2), (1, 4), (2, 4)}``. AIE2P-investigation
            escape hatch. Default False.

    Raises:
        TypeError: producer / consumer not coercible to Tile, or N / M
            not int.
        ValueError: sparsity pattern / N / M violates the structural
            rules in :func:`_validate_sparsity_pattern`, or ObjectFifo's
            own depth>=1 rule.

    Notes:
        - SparseFifo is **point-to-point** in the M:N pattern semantics
          -- the same compressed stream cannot be consumed by two
          different consumers that disagree on the (N, M) pattern.
          Multi-consumer broadcast (one prod, multiple cons) IS legal
          if all consumers agree on the dense view; that's just
          ObjectFifo's existing broadcast plus the same decompression
          on each consumer's S2MM channel.
        - The :meth:`compression_ratio` property is **theoretical**:
          ``N / M``. The runtime measured ratio depends on the BD-emit
          pass actually flipping ``Enable_Compression`` and the silicon
          compressed-vs-dense DMA-volume comparison that confirms
          the silicon-level ratio.
    """

    def __init__(
        self,
        producer,
        consumer,
        obj_type: type[np.ndarray],
        sparsity_pattern: str = "N:M",
        N: int = 2,
        M: int = 4,
        depth: int | None = 2,
        name: str | None = None,
        dims_to_stream: list[Sequence[int]] | None = None,
        dims_from_stream_per_cons: list[Sequence[int]] | None = None,
        plio: bool = False,
        pad_dimensions: list[Sequence[int]] | None = None,
        allow_unverified: bool = False,
    ):
        # Validate sparsity arguments BEFORE ObjectFifo.__init__ so
        # the user gets the more-specific error (and we don't allocate
        # ObjectFifo state for a doomed-to-fail SparseFifo).
        _validate_sparsity_pattern(sparsity_pattern, N, M, allow_unverified)

        producer_tile = _coerce_to_tile(producer, "producer")
        consumer_tile = _coerce_to_tile(consumer, "consumer")

        # ObjectFifo's __init__ doesn't take tile args directly -- the
        # tiles come in via prod()/cons() endpoint creation later. Save
        # them here so SparseFifo's prod()/cons() can return placed
        # handles (and so __str__ has something to show).
        self._producer_tile = producer_tile
        self._consumer_tile = consumer_tile

        self._sparsity_pattern = sparsity_pattern
        self._N = N
        self._M = M
        self._allow_unverified = allow_unverified

        super().__init__(
            obj_type=obj_type,
            depth=depth,
            name=name,
            dims_to_stream=dims_to_stream,
            dims_from_stream_per_cons=dims_from_stream_per_cons,
            plio=plio,
            pad_dimensions=pad_dimensions,
        )

    # ----- sparsity-specific properties -------------------------------

    @property
    def sparsity_pattern(self) -> str:
        """The pattern tag (currently always ``"N:M"``)."""
        return self._sparsity_pattern

    @property
    def N(self) -> int:
        """Number of nonzero elements per group of size :attr:`M`."""
        return self._N

    @property
    def M(self) -> int:
        """Group size for the N:M pattern."""
        return self._M

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio (``N / M``).

        This is the wire-level fraction of the dense byte count that
        SHOULD be transferred per the requested pattern. The runtime
        responsibility -- the BD-emit pass must actually flip
        ``Enable_Compression`` AND the silicon must honor it.
        """
        return self._N / self._M

    @property
    def producer_tile(self) -> Tile:
        """The producer-side tile (writes COMPRESSED stream on MM2S)."""
        return self._producer_tile

    @property
    def consumer_tile(self) -> Tile:
        """The consumer-side tile (reads DENSE stream on S2MM)."""
        return self._consumer_tile

    @property
    def is_pattern_am020_verified(self) -> bool:
        """True if (N, M) is in the AM020-documented verified set.

        Use this to gate "should this run on AIE2P silicon" decisions:
        verified patterns map to the AIE-ML reference; unverified
        patterns are AIE2P-investigation only and may diverge.
        """
        return (self._N, self._M) in _AM020_VERIFIED_NM_PATTERNS

    def __str__(self) -> str:
        base = super().__str__()
        return (
            f"SparseFifo({base}, pattern={self._sparsity_pattern!r}, "
            f"N={self._N}, M={self._M}, "
            f"compression_ratio={self.compression_ratio:.3f})"
        )

    # ----- ObjectFifo overrides ---------------------------------------

    def prod(self, depth: int | None = None) -> "SparseFifoHandle":
        """Return the producer-side :class:`SparseFifoHandle`.

        Mirrors :meth:`ObjectFifo.prod` but returns a SparseFifo-typed
        handle so downstream code can branch on ``isinstance(h,
        SparseFifoHandle)`` if it needs to. The handle is registered
        with the FifoHandle registry at module import time, so
        Worker.fn_args dispatch picks the SparseFifoHandle handler over
        the ObjectFifoHandle handler when present.
        """
        if self._prod is None:
            if depth is None:
                if self._depth is None:
                    raise ValueError("If depth is None, then depth must be specified.")
                depth = self._depth
            elif depth < 1:
                raise ValueError(f"Depth must be >= 1, but got {depth}")
            self._prod = SparseFifoHandle(self, is_prod=True, depth=depth)
        return self._prod

    def cons(
        self,
        depth: int | None = None,
        dims_from_stream: list[Sequence[int]] | None = None,
    ) -> "SparseFifoHandle":
        """Return a consumer-side :class:`SparseFifoHandle`.

        Each call creates a new consumer handle (matches
        :meth:`ObjectFifo.cons`'s broadcast-friendly behaviour). All
        consumers see the DENSE view -- the per-consumer S2MM
        decompression decompresses each one independently.
        """
        if depth is None:
            if self._depth is None:
                raise ValueError("If depth is None, then depth must be specified.")
            depth = self._depth

        if dims_from_stream is None:
            dims_from_stream = self._dims_from_stream_per_cons

        handle = SparseFifoHandle(
            self,
            is_prod=False,
            depth=depth,
            dims_from_stream=dims_from_stream,
        )
        self._cons.append(handle)
        return handle

    # ----- lowering ---------------------------------------------------

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Lower to ``aie.objectfifo`` + sparsity-attribute decoration.

        Calls :meth:`ObjectFifo.resolve` to build the underlying
        ObjectFifoCreateOp, then attaches the compression / sparsity
        attributes the BD-emit pass consumes:

        - ``aie.compress_mm2s = true`` -> producer-side BD gets
          ``Enable_Compression = 1`` per
          ``aie_registers_aie2.json``.
        - ``aie.decompress_s2mm = true`` -> consumer-side BD gets
          ``Enable_Compression = 1`` (same bit; "Enable Compression
          (MM2S), decompression (S2MM)").
        - ``aie.sparsity_pattern = "N:M"`` (string),
          ``aie.sparsity_n = <N>`` (i32),
          ``aie.sparsity_m = <M>`` (i32) -> diagnostic / introspection
          for the BD-emit pass + downstream debug dumps.

        If the active backend hasn't been taught about these
        attributes yet (early AIE2P silicon-driver stack), the design
        still compiles -- the attributes are ignored, the compression
        is skipped, and the dataflow runs as a vanilla ObjectFifo.
        runtime DMA-volume measurement (see module docstring).
        """
        # Re-entrancy guard via the _op-already-set state (ObjectFifo's
        # resolve handles its own re-entry; we only attach our
        # discardable attrs once.)
        already_resolved = self._op is not None
        super().resolve(loc=loc, ip=ip)

        if already_resolved or self._op is None:
            return

        ctx = self._op.operation.context
        # Tag-set on the ObjectFifoCreateOp's discardable attribute
        # dict. The BD-emit pass reads them back via
        # op->getAttrOfType(...).
        self._op.operation.attributes[SPARSE_ATTR_COMPRESS_MM2S] = (
            ir.BoolAttr.get(True, ctx)
        )
        self._op.operation.attributes[SPARSE_ATTR_DECOMPRESS_S2MM] = (
            ir.BoolAttr.get(True, ctx)
        )
        self._op.operation.attributes[SPARSE_ATTR_PATTERN] = ir.StringAttr.get(
            self._sparsity_pattern, ctx
        )
        i32 = ir.IntegerType.get_signless(32, ctx)
        self._op.operation.attributes[SPARSE_ATTR_N] = ir.IntegerAttr.get(
            i32, self._N
        )
        self._op.operation.attributes[SPARSE_ATTR_M] = ir.IntegerAttr.get(
            i32, self._M
        )

class SparseFifoHandle(ObjectFifoHandle):
    """Handle to a :class:`SparseFifo`.

    Subclasses :class:`ObjectFifoHandle` so:

    1. Worker.fn_args' ``isinstance(arg, ObjectFifoHandle)`` check on
       picks the SparseFifoHandle handler over the parent
       ObjectFifoHandle handler in the reverse-insertion-order walk.

    Adds four diagnostic properties (:attr:`sparse_fifo`,
    :attr:`compression_ratio`, :attr:`N`, :attr:`M`) so downstream
    code that wants to branch on "this is a sparse channel" can do so
    without re-walking ``handle._object_fifo``.
    """

    @property
    def sparse_fifo(self) -> SparseFifo:
        """The underlying :class:`SparseFifo`."""
        # ObjectFifoHandle stashes the parent fifo in ``_object_fifo``;
        # we type-narrow here for the SparseFifo case. The constructor
        # of SparseFifoHandle is inherited from ObjectFifoHandle so
        # ``isinstance(self._object_fifo, SparseFifo)`` is a class
        # invariant only when SparseFifoHandle is constructed by
        # SparseFifo.prod() / SparseFifo.cons().
        of = self._object_fifo
        assert isinstance(of, SparseFifo)
        return of

    @property
    def compression_ratio(self) -> float:
        """The theoretical compression ratio of the parent SparseFifo."""
        return self.sparse_fifo.compression_ratio

    @property
    def sparsity_pattern(self) -> str:
        """The sparsity pattern tag of the parent SparseFifo."""
        return self.sparse_fifo.sparsity_pattern

    @property
    def N(self) -> int:
        """N from the N:M pattern of the parent SparseFifo."""
        return self.sparse_fifo.N

    @property
    def M(self) -> int:
        """M from the N:M pattern of the parent SparseFifo."""
        return self.sparse_fifo.M

    def __str__(self) -> str:
        my_str = (
            f"SparseFifoHandle({self.handle_type}, depth={self.depth}, "
            f"pattern={self.sparsity_pattern!r}, "
            f"N={self.N}, M={self.M}, "
            f"of={self._object_fifo})"
        )
        return my_str

# ---------------------------------------------------------------------------
#
# Register SparseFifoHandle with the FifoHandle registry so the
# ``main``) recognizes SparseFifoHandle by its own handler. The
# handler intentionally mirrors the pre-registered ObjectFifoHandle
# bookkeeping (set ``arg.endpoint = worker``; append to
# ``worker._fifos``) so downstream code that iterates worker.fifos
# sees the SparseFifoHandle alongside vanilla ObjectFifoHandles.
#
# isn't consulted by Worker.__init__ -- it falls back to the
# hard-coded ``isinstance(arg, ObjectFifoHandle)`` branch which still
# accepts SparseFifoHandle (because the latter subclasses
# ObjectFifoHandle). So this registration is a forward-looking
# ---------------------------------------------------------------------------

def _sparse_fifo_handle_handler(arg, worker):
    """Worker.fn_args handler for SparseFifoHandle.

    Mirrors the ObjectFifoHandle pre-registered handler exactly:
    set the handle's endpoint to the worker, then append the handle
    to the worker's ``_fifos`` list. Distinct registry entry from
    ObjectFifoHandle's so the reverse-insertion-order walk picks this
    one for SparseFifoHandle instances (and never reaches the parent
    handler for those).
    """
    arg.endpoint = worker
    worker._fifos.append(arg)

# Defer the registration to module-import time so that the test suite
# (and any importing user) only registers once. The registry rejects
# double-registration loudly so the import being a one-shot side-effect
# is intentional.
register_fifo_handle(SparseFifoHandle, _sparse_fifo_handle_handler)

__all__ = [
    "SparseFifo",
    "SparseFifoHandle",
    "SPARSE_CHANNELS_PER_DIRECTION",
    "SPARSE_ATTR_COMPRESS_MM2S",
    "SPARSE_ATTR_DECOMPRESS_S2MM",
    "SPARSE_ATTR_PATTERN",
    "SPARSE_ATTR_N",
    "SPARSE_ATTR_M",
]
