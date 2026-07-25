# program.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import logging

logger = logging.getLogger(__name__)

from ..extras.context import mlir_mod_ctx  # pyright: ignore[reportMissingImports]
from ..helpers.dialects.func import FuncBase
from ..dialects.aie import device

from .device import Device
from .runtime import Runtime
from .scratchpad_parameter import ScratchpadParameter
from .resolvable import Resolvable
from ..utils import trace as trace_utils


class Program:
    def __init__(
        self,
        device: Device,
        rt: Runtime,
        workers: "list | None" = None,
    ):
        """A Program represents all design information needed to run the design on a device.

        !!! note
            MLIR verification (`ctx.module.operation.verify()`) is performed inside
            [`resolve_program`][iron.program.Program.resolve_program], not during construction.

        Args:
            device (Device): The device used to generate the final MLIR for the design.
            rt (Runtime): The runtime object for the design.
            workers (list[Worker] | None, optional): The Workers to run on the
                device. Defaults to None (no workers). Workers are passed here
                explicitly rather than started from within the runtime sequence.
        """
        self._device = device
        self._rt = rt
        self._workers = list(workers) if workers is not None else []
        self._trace_size = None
        self._trace_workers = None
        self._reuse_output_buffer = False
        self._egress_shim_col = 0
        self._coretile_events = None
        self._coremem_events = None
        self._memtile_events = None
        self._shimtile_events = None

    def enable_trace(
        self,
        trace_size: int | None = None,
        workers: list | None = None,
        reuse_output_buffer: bool = False,
        coretile_events: list | None = None,
        coremem_events: list | None = None,
        memtile_events: list | None = None,
        shimtile_events: list | None = None,
        egress_shim_col: int = 0,
    ):
        """Enable hardware tracing for this program.

        Configures the AIE trace units and routes trace packets to DDR via the shim DMA.
        Lives on Program (not Runtime) because it configures both the traced
        Workers' tiles and the Runtime's trace-buffer sequencing.

        Args:
            trace_size (int): Size of the trace buffer in bytes.
            workers (list[Worker] | None, optional): Specific workers to trace. If None,
                all workers with ``trace`` set will be traced. Defaults to None.
            reuse_output_buffer (bool, optional): When False (default), trace
                lowering appends a dedicated trace-buffer argument to the
                runtime_sequence; it lands at the tail so enabling trace never
                perturbs the data arguments' indices. When True, trace data is
                written into the tail of the last output buffer, saving a host
                buffer. Defaults to False.
            coretile_events (list | None, optional): List of up to 8 core tile trace events.
                See [the AIEX dialect reference](../AIEXDialect.md) for available
                events under (type)EventAIE such as CoreEventAIE.
                Defaults to None (uses hardware defaults).
            coremem_events (list | None, optional): List of up to 8 core memory trace events.
                Defaults to None (uses hardware defaults).
            memtile_events (list | None, optional): List of up to 8 mem tile trace events.
                Defaults to None (uses hardware defaults).
            shimtile_events (list | None, optional): List of up to 8 shim tile trace events.
                Defaults to None (uses hardware defaults).
            egress_shim_col (int, optional): Column of the shim tile used to
                egress trace packets to DDR. Defaults to 0.
        """
        self._trace_size = trace_size
        self._trace_workers = workers
        self._reuse_output_buffer = reuse_output_buffer
        self._coretile_events = coretile_events
        self._coremem_events = coremem_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events
        self._egress_shim_col = egress_shim_col

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
            self._device = device_type()  # pyright: ignore[reportCallIssue]

            # Resolve parameters at module scope (before the aie.device).
            # aiex.scratchpad_parameter ops are global across all devices because the
            # scratchpad is a single hardware resource shared by all PDIs.
            for w in self._workers:
                for arg in w.flat_fn_args:
                    if isinstance(arg, ScratchpadParameter):
                        arg.resolve()
            for p in self._rt._scratchpad_parameters:
                p.resolve()

            @device(self._device.resolve(), sym_name=device_name)
            def device_body():
                # Collect all fifos. Runtime-driven fifos already have their shim
                # endpoints bound (Runtime registered its fn_args at construction),
                # so they resolve here with both ends known -- the sequence body
                # itself is emitted LAST (self._rt.resolve() below), after workers,
                # so body verbs that read worker-side state (barrier locks, worker
                # Buffer placement) see it resolved.
                all_fifos = set()
                all_fifos.update(self._rt.fifos)
                for w in self._workers:
                    all_fifos.update(w.fifos)

                # Sort fifos for deterministic resolve
                all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

                # Collect all tiles. Two workers landing on the same compute
                # tile (pinned or after placement) is caught by the aie.device
                # verifier's one-core-per-tile check, so no Python-side guard.
                all_tiles = []
                for w in self._workers:
                    all_tiles.append(w.tile)
                    # Generic: any user-side Resolvable in fn_args may declare
                    # additional tile dependencies via tiles(). Default is [].
                    for arg in w.flat_fn_args:
                        if isinstance(arg, Resolvable):
                            all_tiles.extend(arg.tiles())
                for f in all_fifos:
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

                # Generate fifos
                for f in all_fifos:
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
                    for arg in w.flat_fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        elif isinstance(arg, Resolvable):
                            arg.resolve()

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

                # Generate trace routes
                # TODO Need to iterate over all tiles or workers & fifos to make list of tiles to trace
                #      Alternatively, we merge the mechanism for packet routed objfifos so we use unique
                #      route IDs for trace as well

                # Scan workers and build list of tiles to trace
                tiles_to_trace = []
                if self._trace_workers is not None:
                    for w in self._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in self._workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)
                if self._trace_size is not None and self._trace_size > 0:
                    trace_utils.configure_trace(
                        tiles_to_trace,
                        coretile_events=self._coretile_events,
                        coremem_events=self._coremem_events,
                        memtile_events=self._memtile_events,
                        shimtile_events=self._shimtile_events,
                    )

                # Emit the runtime sequence body LAST: workers, their locks, and
                # worker Buffers are now resolved, so body verbs that read that
                # state (barrier.set, inline_ops over a worker Buffer) are valid.
                # Its shim DMAs reference fifos by symbol name (forward ref), so
                # emitting after the fifo ops is fine.
                self._rt.resolve(
                    trace_size=self._trace_size,
                    reuse_output_buffer=self._reuse_output_buffer,
                    egress_shim_col=self._egress_shim_col,
                )

            self._print_verify(ctx)
            return ctx.module

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if verify != True:
            raise RuntimeError(f"MLIR module failed verification: {verify}")
