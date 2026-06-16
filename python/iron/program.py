# program.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import logging

logger = logging.getLogger(__name__)

from ..extras.context import mlir_mod_ctx  # type: ignore
from ..helpers.dialects.func import FuncBase
from ..dialects.aie import device

from .device import Device
from .runtime import Runtime
from .dataflow import ObjectFifoHandle
from .scratchpad_parameter import ScratchpadParameter
from .resolvable import Resolvable
from ..utils import trace as trace_utils


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
        rt: Runtime,
        workers: list | None = None,
    ):
        """A Program represents all design information needed to run the design on a device.

        Note: MLIR verification (``ctx.module.operation.verify()``) is performed inside
        :meth:`resolve_program`, not during construction.

        Args:
            device (Device): The device used to generate the final MLIR for the design.
            rt (Runtime): The runtime object for the design.
            workers (list[Worker] | None, optional): The Workers to place and run.
                The runtime sequence references their fifos; the Program resolves
                their tiles, cores, and fifos. Defaults to None (no Workers — a
                pure host-driven data-movement design).
        """
        self._device = device
        self._rt = rt
        self._workers = list(workers) if workers is not None else []
        # Let the runtime discover the fifos it touches (by walking the fifo
        # link graph from the Workers) so they resolve in a deterministic order.
        self._rt.discover_fifos(self._workers)

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

            # Resolve parameters at module scope (before the aie.device).
            # aiex.scratchpad_parameter ops are global across all devices because the
            # scratchpad is a single hardware resource shared by all PDIs.
            for w in self._workers:
                for arg in w.fn_args:
                    if isinstance(arg, ScratchpadParameter):
                        arg.resolve()
            for p in self._rt._scratchpad_parameters:
                p.resolve()

            @device(self._device.resolve(), sym_name=device_name)
            def device_body():
                # Collect all fifos
                all_fifos = set()
                all_fifos.update(self._rt.fifos)
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
                # Lower-level: explicit Flow / TileDma / Lock primitives
                # contribute tiles too.
                for fl in self._rt.flows:
                    all_tiles.extend(fl.all_tiles())
                for td in self._rt.tile_dmas:
                    all_tiles.extend(td.all_tiles())
                for lk in self._rt.locks:
                    all_tiles.append(lk.tile)

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
                for fl in self._rt.flows:
                    fl.resolve()

                # Generate explicit Locks (must come before TileDma + Worker
                # bodies that reference them; Buffers attached to worker
                # fn_args are still resolved in the worker loop below).
                for lk in self._rt.locks:
                    lk.resolve()

                # Resolve any Buffers referenced by explicit TileDma programs
                # (those aren't reached via worker.fn_args).
                for td in self._rt.tile_dmas:
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

                # Generate trace routes
                # TODO Need to iterate over all tiles or workers & fifos to make list of tiles to trace
                #      Alternatively, we merge the mechanism for packet routed objfifos so we use unique
                #      route IDs for trace as well

                # Scan workers and build list of tiles to trace
                tiles_to_trace = []
                if self._rt._trace_workers is not None:
                    for w in self._rt._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in self._workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)
                if self._rt._trace_size is not None and self._rt._trace_size > 0:
                    trace_utils.configure_trace(
                        tiles_to_trace,
                        coretile_events=self._rt._coretile_events,
                        coremem_events=self._rt._coremem_events,
                        memtile_events=self._rt._memtile_events,
                        shimtile_events=self._rt._shimtile_events,
                    )

                # Create worker barrier locks before the runtime sequence so
                # rt.set_barrier(...) in the sequence body can emit set_lock
                # against them (the sequence now resolves ahead of the cores).
                for w in self._workers:
                    w.resolve_barrier_locks()

                # In/Out Sequence. The body emits DMA ops referencing fifos by
                # symbol name, then creates the runtime-touched fifo ops at device
                # scope once their endpoints are known. Emitted before the cores
                # so those fifo ops exist by the time the cores reference them.
                self._rt.resolve(device=self._device)

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
                for td in self._rt.tile_dmas:
                    td.resolve()

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            raise RuntimeError(f"MLIR module failed verification: {verify}")
