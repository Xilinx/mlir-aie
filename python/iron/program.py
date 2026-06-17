# program.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import logging

logger = logging.getLogger(__name__)

import numpy as np

from ..extras.context import mlir_mod_ctx  # type: ignore
from ..helpers.dialects.func import FuncBase
from ..dialects.aie import device

from .. import ir  # type: ignore
from .device import Device
from .dataflow import ObjectFifoHandle
from .scratchpad_parameter import ScratchpadParameter
from .resolvable import Resolvable
from .runtime.data import RuntimeData
from .runtime._sequence import discover_fifos, resolve_sequence
from ..utils import trace as trace_utils


def _classify_sequence_args(arg_types: list) -> list:
    """Turn the sequence arg-type list into per-arg items for resolution.

    A tensor type (an ``np.ndarray`` parameterization) becomes a
    :class:`RuntimeData` handle — it carries shape/tap and is passed into the
    body so verbs like ``of.prod().fill(A)`` can read it. A scalar type (an
    ``ir.Type`` or a zero-arg callable producing one) is kept as-is: the body
    receives the raw SSA ``Value`` for it, so no wrapper object is needed.
    """
    items = []
    for t in arg_types:
        if getattr(t, "__origin__", None) is np.ndarray:
            items.append(RuntimeData(t))
        elif isinstance(t, ir.Type) or callable(t):
            items.append(t)  # scalar: a bare MLIR type (or callable producing one)
        else:
            raise TypeError(
                f"Unsupported sequence argument type: {type(t).__name__}. "
                f"Expected np.ndarray type, ir.Type, or callable."
            )
    return items


def _fifo_endpoints_ready(handle, _seen=None) -> bool:
    """Whether a fifo's endpoints are fully established for eager resolution.

    A fifo touched by the runtime gets its shim-side endpoint — and, for a
    runtime consumer, the cons handle itself — only when the sequence body issues
    its fill/drain (which runs after the Program's up-front collection). Until
    then the fifo is not ready; the verb resolves it lazily on first touch.

    Readiness is transitive across ``ObjectFifoLink`` (split/join/forward):
    resolving a fifo cascades to the fifos it is linked with, so a worker-side
    fifo (e.g. ``memA``) linked to a runtime-fed input (``inA``) is only ready
    once that input's shim endpoint is set. Such a group is resolved together,
    lazily, when the runtime touches it.
    """
    # Imported here to avoid a module-load cycle with dataflow.
    from .dataflow.objectfifo import ObjectFifoLink

    of = handle._object_fifo
    if _seen is None:
        _seen = set()
    if id(of) in _seen:
        return True
    _seen.add(id(of))

    if of._prod is None or of._prod.endpoint is None:
        return False
    if len(of._cons) < 1:
        return False
    if not all(con.endpoint is not None for con in of._cons):
        return False

    # Follow links to peers; the group resolves as a unit.
    endpoints = [of._prod.endpoint] + [con.endpoint for con in of._cons]
    for endpoint in endpoints:
        if isinstance(endpoint, ObjectFifoLink):
            for peer in (*endpoint._srcs, *endpoint._dsts):
                if not _fifo_endpoints_ready(peer, _seen):
                    return False
    return True


class Program:
    def __init__(
        self,
        device: Device,
        sequence,
        arg_types: list | None = None,
        workers: list | None = None,
        flows: list | None = None,
        tile_dmas: list | None = None,
        trace_tiles: list | None = None,
        trace: "TraceBuffer | None" = None,
    ):
        """All design information needed to run a design on a device.

        The host-side control program is supplied directly as ``sequence`` — a
        plain callable, like a :class:`Worker`'s ``core_fn``. It runs once,
        inside the ``aie.runtime_sequence`` builder, during
        :meth:`resolve_program`; ordinary Python control flow (``range_`` /
        ``if_``) and arithmetic on its scalar arguments lower to ``scf`` /
        ``arith`` ops in place. Inside it, drive data movement on the objects
        themselves — ``of.prod().fill(...)``, ``of.cons().drain(...)`` — and
        group completions with :class:`TaskGroup`.

        Device-scope structure is mostly *inferred*, not listed: locks, buffers,
        and scratchpad parameters are discovered from the ``workers`` (their
        ``fn_args``) and ``tile_dmas`` (their BD acquire/release + buffers). Only
        the two routing *roots* that nothing else references — explicit
        ``flows`` and ``tile_dmas`` — are passed, and only the handful of
        lower-level explicit-routing designs need them; high-level ObjectFifo
        designs pass neither.

        Tracing has two parts: *sources* (what to capture) and the *sink* (where
        bytes land). Per-Worker tracing is declared via
        ``Worker(trace=TileTrace(...))``; non-worker tiles via ``trace_tiles``;
        the sink is the ``trace`` :class:`TraceBuffer`.

        Note: MLIR verification is performed inside :meth:`resolve_program`.

        Args:
            device (Device): The device used to generate the final MLIR.
            sequence (Callable): The runtime-sequence body. Called once with one
                argument per entry in ``arg_types``, in order.
            arg_types (list | None): One type per sequence argument — an
                ``np.ndarray`` type for a buffer, or an MLIR scalar type (e.g.
                ``T.i32``) / zero-arg callable producing one for a runtime scalar.
                Defaults to None (a sequence taking no arguments).
            workers (list[Worker] | None): The Workers to place and run.
            flows (list[Flow] | None): Explicit AXI-stream routes (lower-level
                designs only). Defaults to None.
            tile_dmas (list[TileDma] | None): Explicit per-tile DMA programs
                (lower-level designs only). Their locks/buffers are inferred.
                Defaults to None.
            trace_tiles (list[TileTrace] | None): Trace sources for tiles not
                owned by a Worker. Each must carry an explicit ``tile``.
            trace (TraceBuffer | None): The trace output buffer (sink).
        """
        self._device = device
        self._sequence_fn = sequence
        self._sequence_items = _classify_sequence_args(arg_types or [])
        self._workers = list(workers) if workers is not None else []
        self._flows = list(flows) if flows is not None else []
        self._tile_dmas = list(tile_dmas) if tile_dmas is not None else []
        self._trace_tiles = list(trace_tiles) if trace_tiles is not None else []
        self._trace = trace
        # Fifos the sequence will touch, discovered by walking the link graph
        # from the Workers, so they resolve in a deterministic order.
        self._fifos = discover_fifos(self._workers)
        # Locks / buffers / scratchpad params are inferred during resolution
        # from workers + tile_dmas (see _collect_inferred_*).

    def resolve_program(self, device_name="main"):
        """This method resolves the program components in order to generate MLIR.

        Tiles are emitted as aie.logical_tile ops. The --aie-place-tiles pass
        in the compilation pipeline converts them to aie.tile ops.

        Returns:
            module (Module): The module containing the MLIR context information.
        """
        with mlir_mod_ctx() as ctx:
            # Create a fresh device instance of the same type to avoid stale MLIR operations
            # This preserves the device configuration while ensuring clean state
            device_type = type(self._device)
            # For dynamically created device classes, the constructor takes no arguments
            self._device = device_type()

            # Resolve scratchpad parameters at module scope (before the
            # aie.device): they are global across all devices because the
            # scratchpad is a single hardware resource shared by all PDIs.
            # Inferred from the workers' fn_args — no separate registration.
            for p in self._inferred_scratchpad_parameters():
                p.resolve()

            @device(self._device.resolve(), sym_name=device_name)
            def device_body():
                # Collect all fifos
                all_fifos = set(self._fifos)
                for w in self._workers:
                    all_fifos.update(w.fifos)

                # Sort fifos for deterministic resolve
                all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

                # Collect all tiles, validating no two workers share the same coordinates
                all_tiles = []
                worker_tile_coords = set()
                for w in self._workers:
                    if w.tile.col is not None and w.tile.row is not None:
                        coord = (w.tile.col, w.tile.row)
                        if coord in worker_tile_coords:
                            raise ValueError(
                                f"Multiple workers cannot share the same tile: {w.tile}"
                            )
                        worker_tile_coords.add(coord)
                    all_tiles.append(w.tile)
                    # Generic: any user-side Resolvable in fn_args may declare
                    # additional tile dependencies via tiles(). Default is [].
                    for arg in w.fn_args:
                        if isinstance(arg, Resolvable):
                            all_tiles.extend(arg.tiles())
                for f in all_fifos:
                    # A fifo touched by the runtime gets its shim-side endpoint
                    # (and, for a runtime consumer, its cons handle) only when the
                    # sequence body issues the fill/drain — which runs after this
                    # collection. Such a fifo is resolved lazily on first verb
                    # touch, so skip it here entirely.
                    if not _fifo_endpoints_ready(f):
                        continue
                    all_tiles.extend([e.tile for e in f.all_of_endpoints()])
                    # Shared-memory delegate tile (ObjectFifo.delegate_tile kwarg)
                    # may not appear in any prod/cons endpoint, so pick it up
                    # explicitly so resolve_tile() runs on it before fifo resolution.
                    if f._object_fifo._delegate_tile is not None:
                        all_tiles.append(f._object_fifo._delegate_tile)
                # Lower-level: explicit Flow / TileDma primitives contribute
                # tiles too. Locks are inferred (from tile_dmas), and their
                # tiles are already covered by the tile_dma tiles below.
                for fl in self._flows:
                    all_tiles.extend(fl.all_tiles())
                for td in self._tile_dmas:
                    all_tiles.extend(td.all_tiles())
                for lk in self._inferred_locks():
                    all_tiles.append(lk.tile)
                # Non-worker tiles named for tracing must be placed too.
                for tt in self._trace_tiles:
                    if tt.tile is not None:
                        all_tiles.append(tt.tile)

                # Resolve tiles
                for t in all_tiles:
                    self._device.resolve_tile(t)

                # Generate fifos. A fifo whose runtime-facing endpoint is not yet
                # established (its shim side / runtime cons handle is created when
                # the sequence body issues the fill/drain) is resolved lazily at
                # that point instead.
                for f in all_fifos:
                    if not _fifo_endpoints_ready(f):
                        continue
                    f.resolve()

                # Generate explicit Flows (peers of ObjectFifo)
                for fl in self._flows:
                    fl.resolve()

                # Generate explicit Locks (inferred from tile_dmas). Must come
                # before TileDma + Worker bodies that reference them; Buffers
                # attached to worker fn_args are resolved in the worker loop below.
                for lk in self._inferred_locks():
                    lk.resolve()

                # Resolve any Buffers referenced by explicit TileDma programs
                # (those aren't reached via worker.fn_args).
                for td in self._tile_dmas:
                    bufs, _ = td.all_buffers_and_locks()
                    for b in bufs:
                        if b.tile is None:
                            b._tile = td.tile
                        b.resolve()

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in self._workers:
                    for arg in w.fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        elif isinstance(arg, ObjectFifoHandle):
                            # A fifo still missing its runtime-facing endpoint is
                            # resolved lazily when the sequence body issues the
                            # fill/drain; resolving it here would read an unset
                            # shim tile.
                            if _fifo_endpoints_ready(arg):
                                arg.resolve()
                        elif isinstance(arg, Resolvable):
                            arg.resolve()

                # Generate trace routes. Each traced Worker contributes its
                # compute tile (via its TileTrace); trace_tiles adds non-worker
                # tiles. The per-unit specs drive the declarative aie.trace ops;
                # -aie-insert-trace-flows assigns packet ids and routes after
                # placement.
                if self._trace is not None:
                    worker_traces = [
                        (w.tile.op, w.trace)
                        for w in self._workers
                        if w.trace is not None
                    ]
                    specs = trace_utils.build_trace_specs(
                        worker_traces, self._trace_tiles
                    )
                    trace_utils.configure_trace_specs(specs)

                # Create worker barrier locks before the runtime sequence so
                # barrier.set(...) in the sequence body can emit set_lock
                # against them (the sequence now resolves ahead of the cores).
                for w in self._workers:
                    w.resolve_barrier_locks()

                # In/Out Sequence. The body emits DMA ops referencing fifos by
                # symbol name, then creates the runtime-touched fifo ops at device
                # scope once their endpoints are known. Emitted before the cores
                # so those fifo ops exist by the time the cores reference them.
                # The trace config (if any) is passed so the sequence emits the
                # trace egress-DMA setup for the single shared buffer.
                resolve_sequence(
                    self._sequence_fn,
                    self._sequence_items,
                    self._device,
                    trace=self._trace,
                )

                # Resolve any remaining worker-side fifos (idempotent) before the
                # cores reference them.
                for f in all_fifos:
                    if _fifo_endpoints_ready(f):
                        f.resolve()

                # Generate core programs
                for w in self._workers:
                    w.resolve()

                # Emit aie.cascade_flow ops for each Worker's outgoing edges.
                # Must run after worker.resolve() so both tiles are placed.
                for w in self._workers:
                    for cf in w._outgoing_cascades:
                        cf.resolve()

                # Generate explicit per-tile DMA programs (lower-level peers
                # of ObjectFifo, paired with Flow + Lock).
                for td in self._tile_dmas:
                    td.resolve()

            self._print_verify(ctx)
            return ctx.module

    # -- inferred device-scope structure --------------------------------------

    def _inferred_scratchpad_parameters(self) -> list:
        """ScratchpadParameters reachable from the workers' fn_args.

        Order-preserving and de-duplicated by identity, so a parameter shared by
        several workers is resolved once.
        """
        seen: list = []
        for w in self._workers:
            for arg in w.fn_args:
                if isinstance(arg, ScratchpadParameter) and arg not in seen:
                    seen.append(arg)
        return seen

    def _inferred_locks(self) -> list:
        """Locks reachable from the explicit tile_dmas' BD acquire/release ops.

        Worker-barrier locks are created separately by ``resolve_barrier_locks``;
        these are the explicit-routing locks a TileDma references. De-duplicated
        by identity.
        """
        seen: list = []
        for td in self._tile_dmas:
            _, locks = td.all_buffers_and_locks()
            for lk in locks:
                if lk not in seen:
                    seen.append(lk)
        return seen

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            raise RuntimeError(f"MLIR module failed verification: {verify}")
