# variable_rate.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON :class:`VariableRateFifo` -- producer-side conditional-forward FIFO.

Promotes the AIE-ML / AIE2P **shared-memory + lock-based** dataflow channel
to a first-class IRON primitive whose *producer* may emit asymmetric
acquire / release counts within a single loop iteration -- i.e. the
producer reads N items from upstream and forwards K <= N to the consumer,
where K is decided at run time.

VariableRateFifo and :class:`PacketFifo` are **siblings**, not
alternatives:

- :class:`PacketFifo` solves "many independent producers fan into one
  consumer at runtime-decided rates" via the AXI stream switch +
  per-packet routing headers (no shared memory; no locks).
- :class:`VariableRateFifo` solves "one producer with a conditional
  forward inside its inner loop" via the existing shared-memory + lock
  ObjectFifo runtime, simply lifting the static-rate invariant the
  loop-unrolling pass otherwise imposes.

Architectural references
------------------------

- AM020 Ch. 1 p. 6 (compute-tile DM + lock units; shared-memory
  dataflow basis for ObjectFifo).
- AM020 Ch. 5 p. 74 (memtile out-of-order BD processing — the
  hardware capability that makes producer-side conditional advance
  safe; the BD that's not fired this iteration simply doesn't get
  scheduled).
  early — the canonical use case: ~12.5% of windows pass NGG, so
  forwarding only those saves ~7x DMA + ~7x consumer cycles).
  variable-rate ObjectFifo investigation).
- ``python/iron/packet.py`` (PacketFifo — sibling primitive; the

User-facing surface
-------------------

.. code-block:: python

    from aie.iron import VariableRateFifo, ObjectFifo, Worker
    from aie.iron.device import Tile
    import numpy as np

    # Upstream (fixed-rate input from shim DMA) -> Tile (conditional
    # forward) -> downstream (variable-rate consumer reads only the
    # forwarded windows).
    in_fifo = ObjectFifo(window_type, name="in", depth=2)
    out_fifo = VariableRateFifo(window_type, name="out", depth=2)

    def filter_kernel(in_handle, out_handle, predicate_fn):
        # Read every input window (fixed-rate upstream).
        in_view = in_handle.acquire(1)
        win = in_view[0]
        if predicate_fn(win):
            # Pass this window to the consumer (one acquire+release pair).
            out_view = out_handle.acquire(1)
            copy(win, out_view[0])
            out_handle.release(1)
        else:
            # Skip: explicitly tell IRON we are NOT forwarding this
            # iteration's slot. No MLIR emitted; this is a marker that
            # the static-rate invariant is intentionally relaxed.
            out_handle.discard(1)
        in_handle.release(1)

    w = Worker(filter_kernel, fn_args=[in_fifo.cons(), out_fifo.prod(),
                                       predicate_fn], tile=Tile(0, 2))

The consumer side is **unchanged** from a vanilla ObjectFifo: the
consumer's ``acquire(1)`` blocks until the producer has actually
released a slot. From the consumer's view, only forwarded windows
are visible.

Why a separate class instead of a flag on ObjectFifo
----------------------------------------------------

ObjectFifo's loop-unrolling lowering (in
``AIEObjectFifoStatefulTransform.cpp::unrollForLoops``) walks the
producer's ``scf.for`` loops and unrolls by
``LCM(objfifo_size_for_each_acquire)`` to match the BD-chain length
to the FIFO depth. Asymmetric acquire/release within an unrolled
iteration breaks the assumption: after unrolling, the conditional
acquire+release pair appears N times in the unrolled body, so the
LCM-unroll math becomes meaningless.

VariableRateFifo opts out of LCM-unrolling for the marked fifos.
The lowering pass instead routes them through the
``dynamicGlobalObjectFifos`` runtime-counter machinery (already
present in the pass; see ``updateGlobalNextIndex`` for the
runtime-counter increment on each release). Runtime counters make
no static-rate assumption; they advance only when a release actually
fires. This is the same machinery the dialect already uses for
``dynamic_lowering`` -annotated tests.

Forcing this onto vanilla ObjectFifo via a constructor flag would
either (a) silently change LCM-unrolling for every existing design,
or (b) require every ObjectFifo user to learn an
"is-this-variable-rate?" invariant. A sibling subclass keeps the
abstraction clean and lets the pass decide based on the fifo's
discardable attr.

Lowering model
--------------

At construction time, :class:`VariableRateFifo` builds a vanilla
:class:`ObjectFifo`. When :meth:`resolve` runs, it pins one
discardable attribute on the lowered ``aie.objectfifo`` op:

- ``aie.variable_rate = true``

The downstream pass change (in
``AIEObjectFifoStatefulTransform.cpp``):

1. **Loop-unrolling exclusion** (``unrollForLoops`` /
   ``checkAndApplyForLoopUnrolling``): the LCM computation skips
   acquire ops whose target ObjectFifoCreateOp carries
   ``aie.variable_rate = true``. The producer's loop is therefore
   not unrolled on the variable-rate fifo's account; if the loop has
   no other (fixed-rate) fifo accesses, the loop is left alone.
2. **Split-fifo attr propagation** (the same propagation slot the
   the consumer-side ObjectFifoCreateOp also carries the marker
   for diagnostics + downstream introspection.
3. **No new ops, no new BD bits, no new lock semantics.** The
   discard(n) semantic is purely a Python-level no-op marker; the
   producer's source code chooses NOT to call acquire/release on
   skipped iterations, and the lowering's runtime counters take care
   of the rest.

If the active backend hasn't been taught about the
``aie.variable_rate`` attr yet (legacy aie-opt builds), the design
still compiles -- the attr is ignored, and the pass falls through
to LCM-unrolling. The user-visible failure mode is loud: the pass
crashes on the asymmetric acquire/release count or the silicon hangs
on a stale lock. The runtime-counter path (which this primitive
depends on) is already present in the pass and exercised by the
dynamic_lowering test suite.

discard(n) semantics
--------------------

:meth:`VariableRateFifoHandle.discard(n)` is a **producer-only**
no-op marker. It documents that this loop iteration is intentionally
NOT forwarding ``n`` slots to the consumer. No MLIR is emitted. The
method exists so:

1. Static analyzers + readers can see the intent (vs. a bare
   ``pass`` or comment).
2. A future pass can audit that every producer loop iteration
   either calls ``release(>=1)`` OR ``discard(>=1)`` on the
   variable-rate fifo (no silent slot leaks).
3. The Python-level invariant ``acquires_total == releases_total
   + discards_total`` is auditable at the topology level.

Consumer handles do not have ``discard()``: from the consumer's
side, every slot is a real forwarded item; there is nothing to
discard.

Use case: CRISPR PAM filter early-out
-------------------------------------

The canonical motivating workload:
20-bp + 3-bp PAM windows stream from the shim into Tile A. Tile A
checks the PAM byte (NGG); ~12.5 % of windows pass. With vanilla
ObjectFifo, Tile A must forward EVERY window (PAM-failing windows
zero-filled) so the per-iteration acquire/release count stays
constant. The match tiles waste ~87.5 % of cycles on
zero-multiplication. With VariableRateFifo, Tile A forwards only
the passing windows; the match tiles see ~7-8x fewer slots and
reclaim ~7x of their previously-wasted cycles.

References
----------

- AM020 Ch. 1 p. 6 (DM + lock units; shared-memory dataflow basis)
- AM020 Ch. 5 p. 74 (memtile out-of-order BD processing)
- ``python/iron/packet.py`` (PacketFifo — sibling primitive)
- ``python/iron/sparse.py`` (SparseFifo — the analogous
  discardable-attr-on-ObjectFifo pattern this primitive copies)
- ``lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp``
  ``unrollForLoops`` / ``dynamicGlobalObjectFifos`` (the lowering
  routines this primitive interacts with)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import ir  # type: ignore

from .dataflow.fifo_handle_registry import register_fifo_handle
from .dataflow.objectfifo import ObjectFifo, ObjectFifoHandle

# Discardable-attr name pinned by VariableRateFifo.resolve() and
# read by AIEObjectFifoStatefulTransform.cpp's unrollForLoops to
# exclude the fifo from the LCM-unroll set, and by
# propagateSparseCompressionAttr-style split-fifo propagation to
# carry the marker through to the consumer-side fifo.
VARIABLE_RATE_ATTR: str = "aie.variable_rate"

class VariableRateFifo(ObjectFifo):
    """ObjectFifo that allows producer-side conditional forward.

    User-facing surface is identical to :class:`ObjectFifo` -- the
    producer pushes data with :meth:`prod` and the consumer pulls
    data with :meth:`cons`. The difference is a single discardable
    attribute pinned on the lowered ``aie.objectfifo`` op:

    - ``aie.variable_rate = true``

    The downstream ``AIEObjectFifoStatefulTransformPass``:

    1. Excludes this fifo from LCM-based loop unrolling on producer
       loops, so the producer can wrap its acquire/release in a
       runtime conditional without the unroll math fighting it.
    2. Routes accesses through the existing
       ``dynamicGlobalObjectFifos`` runtime-counter machinery (which
       makes no static-rate assumption — counters advance only on
       actual releases).
    3. Propagates the marker through the split-fifo path so
       consumer-side fifos also carry the attribute for diagnostics.

    The :meth:`prod` / :meth:`cons` handles are
    :class:`VariableRateFifoHandle` (an :class:`ObjectFifoHandle`
    subclass); they pass through Worker.fn_args' ``isinstance(arg,
    via the explicit registration at module-import time.

    The producer handle adds a :meth:`discard` method that
    documents-without-emitting "this iteration is intentionally not
    forwarding the next ``n`` slots". This is the auditable
    counterpart to "just don't call acquire/release in the
    skip-branch" — readers can grep for ``.discard(`` to find every
    skip site.

    Args:
        obj_type: The element type (numpy ndarray type) the consumer
            sees. Same as ObjectFifo.
        depth: ObjectFifo depth (mirrors ObjectFifo's argument).
            Default 2.
        name: Optional name. Auto-generated if omitted.
        dims_to_stream: Same as ObjectFifo; producer-side data layout.
        dims_from_stream_per_cons: Same as ObjectFifo; per-consumer
            data layout.
        plio: Same as ObjectFifo.
        pad_dimensions: Same as ObjectFifo.

    Raises:
        ValueError: ObjectFifo's own depth >= 1 rule.

    Notes:
        - VariableRateFifo is **point-to-point at the producer side**
          in the variable-rate semantics — only the producer may
          conditionally skip. The consumer side is fixed-rate (the
          consumer always sees real slots, never stale data).
        - Multi-consumer broadcast (one prod, multiple cons) IS
          legal; that's just ObjectFifo's existing broadcast plus the
          variable-rate marker on the producer side. All consumers
          see the same forwarded subset.
        - For N:1 multi-producer fan-in (the OTHER half of
          the AXI stream-switch packet-routing fabric is the right
          mechanism for that use case.
        - The :meth:`is_variable_rate` property always returns True;
          it exists so downstream code can branch on
          ``isinstance(of, VariableRateFifo)`` OR on the property,
          matching :class:`SparseFifo`'s ``is_pattern_am020_verified``
          pattern.
    """

    def __init__(
        self,
        obj_type: type[np.ndarray],
        depth: int | None = 2,
        name: str | None = None,
        dims_to_stream: list[Sequence[int]] | None = None,
        dims_from_stream_per_cons: list[Sequence[int]] | None = None,
        plio: bool = False,
        pad_dimensions: list[Sequence[int]] | None = None,
    ):
        super().__init__(
            obj_type=obj_type,
            depth=depth,
            name=name,
            dims_to_stream=dims_to_stream,
            dims_from_stream_per_cons=dims_from_stream_per_cons,
            plio=plio,
            pad_dimensions=pad_dimensions,
        )
        # Track the per-handle discard tally for Python-level
        # auditing (the topology can assert acquires == releases +
        # discards on the producer side).
        self._discard_count: int = 0

    # ----- variable-rate-specific properties --------------------------

    @property
    def is_variable_rate(self) -> bool:
        """Always True for VariableRateFifo; False for vanilla ObjectFifo.

        Use this for ``isinstance``-free branching when the caller
        only has a fifo handle (handles expose ``self.is_variable_rate``
        too via :class:`VariableRateFifoHandle`).
        """
        return True

    @property
    def discard_count(self) -> int:
        """Number of slots the producer has marked as discarded so far.

        Updated by :meth:`VariableRateFifoHandle.discard`. Diagnostic
        only; the lowering does not depend on this counter.
        """
        return self._discard_count

    def __str__(self) -> str:
        base = super().__str__()
        return f"VariableRateFifo({base}, discards={self._discard_count})"

    # ----- ObjectFifo overrides ---------------------------------------

    def prod(self, depth: int | None = None) -> "VariableRateFifoHandle":
        """Return the producer-side :class:`VariableRateFifoHandle`.

        Mirrors :meth:`ObjectFifo.prod` but returns a VariableRateFifo
        -typed handle so callers can branch on ``isinstance(h,
        VariableRateFifoHandle)`` if they need to. The producer
        handle exposes :meth:`VariableRateFifoHandle.discard` for the
        skip-this-iteration marker.
        """
        if self._prod is None:
            if depth is None:
                if self._depth is None:
                    raise ValueError(
                        "If depth is None, then depth must be specified."
                    )
                depth = self._depth
            elif depth < 1:
                raise ValueError(f"Depth must be >= 1, but got {depth}")
            self._prod = VariableRateFifoHandle(self, is_prod=True, depth=depth)
        return self._prod

    def cons(
        self,
        depth: int | None = None,
        dims_from_stream: list[Sequence[int]] | None = None,
    ) -> "VariableRateFifoHandle":
        """Return a consumer-side :class:`VariableRateFifoHandle`.

        Each call creates a new consumer handle (matches
        :meth:`ObjectFifo.cons`'s broadcast-friendly behaviour). All
        consumers see the same forwarded subset of slots.
        """
        if depth is None:
            if self._depth is None:
                raise ValueError(
                    "If depth is None, then depth must be specified."
                )
            depth = self._depth

        if dims_from_stream is None:
            dims_from_stream = self._dims_from_stream_per_cons

        handle = VariableRateFifoHandle(
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
        """Lower to ``aie.objectfifo`` + variable-rate-attribute decoration.

        Calls :meth:`ObjectFifo.resolve` to build the underlying
        ObjectFifoCreateOp, then attaches a single boolean
        discardable attribute the lowering pass consumes:

        - ``aie.variable_rate = true`` -> excluded from LCM-based
          loop unrolling on producer loops; routed through
          ``dynamicGlobalObjectFifos`` runtime-counter machinery; the
          marker is propagated through split-fifo paths to the
          consumer-side fifo too (mirrors the SparseFifo

        If the active backend hasn't been taught about the attr (a
        compiles -- the attr is ignored and the pass falls through
        to LCM-unrolling. That fallback is loud (the producer's
        asymmetric acquire/release will trip the static-rate
        assumption); it is NOT a silent precision drift like the
        SparseFifo case. Users will see a pass crash or a silicon
        wedge before incorrect output.
        """
        already_resolved = self._op is not None
        super().resolve(loc=loc, ip=ip)

        if already_resolved or self._op is None:
            return

        ctx = self._op.operation.context
        self._op.operation.attributes[VARIABLE_RATE_ATTR] = ir.BoolAttr.get(
            True, ctx
        )

class VariableRateFifoHandle(ObjectFifoHandle):
    """Handle to a :class:`VariableRateFifo`.

    Subclasses :class:`ObjectFifoHandle` so:

    1. Worker.fn_args' ``isinstance(arg, ObjectFifoHandle)`` check on
       ``main`` accepts it without modification.
       :class:`VariableRateFifoHandle` handler over the parent
       :class:`ObjectFifoHandle` handler in the reverse-insertion-
       order walk.

    Producer handles add :meth:`discard` for the skip-this-iteration
    marker. Consumer handles do not have ``discard()`` -- from the
    consumer's side, every slot is a real forwarded item.

    The :meth:`acquire` / :meth:`release` methods are inherited
    unchanged from :class:`ObjectFifoHandle`. The variable-rate
    semantics are entirely a property of the fifo (the
    ``aie.variable_rate`` attr) plus the producer-side
    discard-or-forward branching pattern in user code.
    """

    @property
    def variable_rate_fifo(self) -> VariableRateFifo:
        """The underlying :class:`VariableRateFifo`."""
        of = self._object_fifo
        assert isinstance(of, VariableRateFifo)
        return of

    @property
    def is_variable_rate(self) -> bool:
        """Always True; convenience property mirroring the parent fifo."""
        return True

    def discard(self, num_elem: int = 1) -> None:
        """Producer-only: mark ``num_elem`` slots as intentionally not forwarded.

        This is a **no-op at the MLIR layer** -- nothing is emitted.
        The method exists so the producer's source code documents the
        skip site explicitly (vs. a bare ``pass`` or a comment).
        Internally the discard count is incremented on the parent
        fifo so static-analysis / topology-validation passes can
        assert the per-producer invariant
        ``acquires_total == releases_total + discards_total``.

        The actual variable-rate semantics are realised by the user's
        kernel function:

        - On forwarded iterations: call ``acquire(n)`` + write data
          + ``release(n)`` on the producer handle.
        - On skipped iterations: call ``discard(n)`` on the producer
          handle (no acquire / release on the variable-rate fifo at
          all). Upstream fifos still need their normal acquire +
          release; only the variable-rate fifo's slot is skipped.

        The lowering pass relies on the
        ``aie.variable_rate = true`` discardable attr (pinned at
        :meth:`VariableRateFifo.resolve`) to opt out of LCM-based
        loop unrolling and route accesses through the
        runtime-counter machinery. See module docstring.

        Args:
            num_elem: Number of slots to mark as skipped this
                iteration. Must be >= 1.

        Raises:
            ValueError: This handle is a consumer (consumers do not
                discard), or ``num_elem < 1``, or
                ``num_elem > self.depth``.
        """
        if not self._is_prod:
            raise ValueError(
                f"VariableRateFifoHandle({self.name}): discard() is "
                f"producer-only; got cons handle"
            )
        if num_elem < 1:
            raise ValueError(
                f"VariableRateFifoHandle({self.name}): discard() requires "
                f"num_elem >= 1, got {num_elem}"
            )
        if num_elem > self._depth:
            raise ValueError(
                f"VariableRateFifoHandle({self.name}): discard({num_elem}) "
                f"exceeds depth {self._depth}"
            )
        # Bump the parent-fifo's discard counter so topology audits
        # can verify the producer-side invariant. This is purely
        # diagnostic; the lowering pass doesn't read it.
        of = self.variable_rate_fifo
        of._discard_count += num_elem

    def __str__(self) -> str:
        return (
            f"VariableRateFifoHandle({self.handle_type}, depth={self.depth}, "
            f"of={self._object_fifo})"
        )

# ---------------------------------------------------------------------------
#
# Register VariableRateFifoHandle with the FifoHandle registry so the
# extensible ``Worker.__init__`` dispatch recognizes it via its own
# handler. The handler intentionally mirrors the pre-registered
# ObjectFifoHandle bookkeeping (set ``arg.endpoint = worker``;
# append to ``worker._fifos``) so downstream code that iterates
# ``worker.fifos`` sees the VariableRateFifoHandle alongside vanilla
# ObjectFifoHandles + SparseFifoHandles + PacketFifoHandles.
#
# consulted by Worker.__init__ -- it falls back to the hard-coded
# ``isinstance(arg, ObjectFifoHandle)`` branch which still accepts
# VariableRateFifoHandle (because the latter subclasses
# ObjectFifoHandle). So this registration is a forward-looking no-op
# ---------------------------------------------------------------------------

def _variable_rate_fifo_handle_handler(arg, worker):
    """Worker.fn_args handler for VariableRateFifoHandle.

    Mirrors the ObjectFifoHandle pre-registered handler exactly:
    set the handle's endpoint to the worker, then append the handle
    to the worker's ``_fifos`` list. Distinct registry entry from
    ObjectFifoHandle's so the reverse-insertion-order walk picks
    this one for VariableRateFifoHandle instances.
    """
    arg.endpoint = worker
    worker._fifos.append(arg)

# Defer registration to module-import time so the test suite (and any
# importing user) only registers once. The registry rejects double-
# registration loudly so the import being a one-shot side-effect is
# intentional.
register_fifo_handle(VariableRateFifoHandle, _variable_rate_fifo_handle_handler)

__all__ = [
    "VariableRateFifo",
    "VariableRateFifoHandle",
    "VARIABLE_RATE_ATTR",
]
