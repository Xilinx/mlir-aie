# program.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import itertools
import logging

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import (
    TraceMode,  # pyright: ignore[reportAttributeAccessIssue]
    device,
)
from ..extras.context import mlir_mod_ctx  # pyright: ignore[reportMissingImports]
from ..helpers.dialects.func import FuncBase
from ..helpers.errors import design_error
from ..helpers.sourceloc import SourceSite
from ..utils import trace as trace_utils
from ..utils.compile.jit.context import get_compile_arg
from .configuration import Configuration
from .dataflow.objectfifo import ObjectFifoLink
from .device import Device
from .resolvable import Resolvable
from .runtime import Runtime
from .scratchpad_parameter import ScratchpadParameter

logger = logging.getLogger(__name__)


class Program:
    def __init__(
        self,
        device: Device | None,
        rt: Runtime,
        workers: "list | None" = None,
    ):
        """Construct a Program with all design information needed to run the design on a device.

        !!! note
            MLIR verification (`ctx.module.operation.verify()`) is performed inside
            [`resolve_program`][iron.program.Program.resolve_program], not during construction.

        Args:
            device (Device): The device used to generate the final MLIR for the design.
                Accepts the ``Device | None`` returned by ``iron.get_current_device``
                directly and raises if no device has been selected, so callers need
                not narrow it first.
            rt (Runtime): The runtime object for the design.
            workers (list[Worker] | None, optional): The Workers to run on the
                device. Defaults to None (no workers). Workers are passed here
                explicitly rather than started from within the runtime sequence.

        Raises:
            ValueError: If ``device`` is None (no NPU device was selected/detected).
        """
        if device is None:
            raise ValueError(
                "Program requires a device, but none was selected. Pass an explicit "
                "Device, or ensure an NPU runtime is available for "
                "iron.get_current_device()."
            )
        configuration = Configuration(
            "main", device, workers=workers or (), runtimes=[rt]
        )
        self._configurations = [configuration]
        self._entry = rt
        self._implicit_configuration = True
        self._site = SourceSite.capture()

    @classmethod
    def compose(
        cls,
        configurations: "list[Configuration]",
        *,
        entry: Runtime,
    ) -> "Program":
        program = cls.__new__(cls)
        program._configurations = list(configurations)
        program._entry = entry
        program._implicit_configuration = False
        program._site = SourceSite.capture()
        program._validate_composition()
        return program

    def _validate_composition(self) -> None:
        if not self._configurations:
            raise ValueError("Program requires at least one configuration.")
        names = [configuration.name for configuration in self._configurations]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"Program has duplicate configuration names: {duplicates}.")
        entry_configuration = self._entry.configuration
        if entry_configuration not in self._configurations:
            raise ValueError(
                f"Entry runtime {self._entry.name!r} does not belong to this Program."
            )

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
        core_trace_mode=TraceMode.EventTime,
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
            core_trace_mode (TraceMode, optional): Trace mode for core tiles.
                Defaults to Event-Time.
        """
        self._entry.configuration.enable_trace(
            trace_size=trace_size,
            workers=workers,
            reuse_output_buffer=reuse_output_buffer,
            coretile_events=coretile_events,
            coremem_events=coremem_events,
            memtile_events=memtile_events,
            shimtile_events=shimtile_events,
            egress_shim_col=egress_shim_col,
            core_trace_mode=core_trace_mode,
        )

    def resolve_program(self, device_name="main"):
        """Resolve the program components in order to generate MLIR.

        Tiles are emitted as aie.logical_tile ops. The --aie-place-tiles pass
        in the compilation pipeline converts them to aie.tile ops.

        Returns:
            module (Module): The module containing the MLIR context information.
        """
        try:
            return self._resolve_program(device_name)
        except Exception as exc:
            raise design_error(exc)

    def _resolve_program(self, device_name):
        context = ir.Context()
        with context, ir.Location.unknown():
            loc = self._site.location()

        self._validate_composition()
        for configuration in self._configurations:
            self._name_unnamed(configuration)
        with mlir_mod_ctx(context=context, location=loc) as ctx:
            for configuration in self._configurations:
                for worker in configuration.workers:
                    for arg in worker.flat_fn_args:
                        if isinstance(arg, ScratchpadParameter):
                            arg.resolve()

            for configuration in self._configurations:
                symbol = (
                    device_name
                    if self._implicit_configuration
                    else configuration.name
                )
                self._resolve_configuration(configuration, symbol, loc)

            if not self._implicit_configuration:
                entry_configuration = self._entry.configuration
                ctx.module.operation.attributes["iron.entry"] = ir.StringAttr.get(
                    f"{entry_configuration.name}:{self._entry.name}"
                )
                ctx.module.operation.attributes["iron.configuration_count"] = (
                    ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), len(self._configurations)
                    )
                )

            self._print_verify(ctx)
            return ctx.module

    def _resolve_configuration(self, configuration, device_name, loc) -> None:
        device_type = type(configuration.device)
        configuration._device = device_type()  # pyright: ignore[reportCallIssue]
        current_device = configuration.device
        workers = configuration.workers
        runtimes = configuration.runtimes

        @device(current_device.resolve(), sym_name=device_name, loc=loc)
        def device_body():
                # Collect all fifos. Runtime-driven fifos already have their shim
                # endpoints bound (Runtime registered its fn_args at construction),
                # so they resolve here with both ends known -- the sequence body
                # is emitted after workers,
                # so body verbs that read worker-side state (barrier locks, worker
                # Buffer placement) see it resolved.
                all_fifos = set()
                for runtime in runtimes:
                    all_fifos.update(runtime.fifos)
                for w in workers:
                    all_fifos.update(w.fifos)

                # Sort fifos for deterministic resolve
                all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

                # Collect all tiles. Two workers landing on the same compute
                # tile (pinned or after placement) is caught by the aie.device
                # verifier's one-core-per-tile check, so no Python-side guard.
                all_tiles = []
                for w in workers:
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
                for fl in configuration.flows:
                    all_tiles.extend(fl.all_tiles())
                for td in configuration.tile_dmas:
                    all_tiles.extend(td.all_tiles())
                for lk in configuration.locks:
                    all_tiles.append(lk.tile)

                # Resolve tiles
                for t in all_tiles:
                    current_device.resolve_tile(t)

                # Generate fifos
                for f in all_fifos:
                    f.resolve()

                # Generate explicit Locks (must come before TileDma + Worker
                # bodies that reference them; Buffers attached to worker
                # fn_args are still resolved in the worker loop below).
                for lk in configuration.locks:
                    lk.resolve()

                # Resolve any Buffers and Locks referenced by explicit TileDma
                # programs (those aren't reached via worker.fn_args).
                for td in configuration.tile_dmas:
                    bufs, locks = td.all_buffers_and_locks()
                    for lk in locks:
                        lk.resolve()
                    for b in bufs:
                        b.place(td.tile)
                        b.resolve()

                # generate functions - this may call resolve() more than once on the same fifo, but that's ok
                for w in workers:
                    for arg in w.flat_fn_args:
                        if isinstance(arg, FuncBase):
                            arg.emit()
                        elif isinstance(arg, Resolvable):
                            if (
                                arg not in configuration.flows
                                and arg not in configuration.tile_dmas
                            ):
                                arg.resolve()

                # Generate core programs
                for w in workers:
                    w.resolve()

                # Emit aie.cascade_flow ops for each Worker's outgoing edges.
                # Must run after worker.resolve() so both tiles are placed.
                for w in workers:
                    for cf in w._outgoing_cascades:
                        cf.resolve()

                # Generate explicit per-tile DMA programs (lower-level peers
                # of ObjectFifo, paired with Flow + Lock).
                configuration.resolve_tile_dmas()

                # Generate trace routes
                # TODO Need to iterate over all tiles or workers & fifos to make list of tiles to trace
                #      Alternatively, we merge the mechanism for packet routed objfifos so we use unique
                #      route IDs for trace as well

                # Scan workers and build list of tiles to trace
                tiles_to_trace = []
                if configuration._trace_workers is not None:
                    for w in configuration._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)
                if (
                    configuration._trace_size is not None
                    and configuration._trace_size > 0
                ):
                    trace_utils.configure_trace(
                        tiles_to_trace,
                        coretile_events=configuration._coretile_events,
                        coremem_events=configuration._coremem_events,
                        memtile_events=configuration._memtile_events,
                        shimtile_events=configuration._shimtile_events,
                        core_trace_mode=configuration._core_trace_mode,
                    )

                # Emit the runtime sequence body after workers, their locks, and
                # worker Buffers are resolved, so body verbs that read that
                # state (barrier.set, inline_ops over a worker Buffer) are valid.
                # Its shim DMAs reference fifos by symbol name (forward ref), so
                # emitting after the fifo ops is fine.
                #
                # On the full-ELF path the runtime sequence must load its own
                # PDI (no xclbin configures the device), so pass the device
                # symbol as the load_pdi reference. The flag is injected into
                # the compile context by CompilableDesign.
                for runtime in runtimes:
                    load_pdi_device_ref = (
                        device_name
                        if runtime is self._entry
                        and get_compile_arg("_iron_full_elf")
                        else None
                    )
                    runtime.resolve(
                        trace_size=configuration._trace_size,
                        reuse_output_buffer=configuration._reuse_output_buffer,
                        egress_shim_col=configuration._egress_shim_col,
                        load_pdi_device_ref=load_pdi_device_ref,
                        device=current_device,
                    )

                # Flow transfers name their allocations while the sequence runs.
                # Resolve both at device scope using those symbol references.
                for fl in configuration.flows:
                    fl.resolve()

        for runtime in runtimes:
            for p in runtime.scratchpad_parameters:
                p.resolve()

    def _name_unnamed(self, configuration: Configuration) -> None:
        """Name the ObjectFifos, and the Buffers the runtime writes, left unnamed.

        Ops refer to them by symbol, so they are numbered here in the order the
        design reaches them, independent of what else the process has built.
        """
        fifos = [
            h._object_fifo
            for runtime in configuration.runtimes
            for h in runtime.fifos
        ]
        fifos += [h._object_fifo for w in configuration.workers for h in w.fifos]
        fifos = list(dict.fromkeys(fifos))
        for of in fifos:
            for handle in [of._prod, *of._cons]:
                link = handle.endpoint if handle is not None else None
                if isinstance(link, ObjectFifoLink):
                    reached = [h._object_fifo for h in [*link._srcs, *link._dsts]]
                    fifos += [f for f in reached if f not in fifos]
        rtps = [
            b for w in configuration.workers for b in w.buffers if b._use_write_rtp
        ]
        taken = {of.name for of in fifos} | {b._name for b in rtps}

        def fresh(prefix):
            name = next(
                n for i in itertools.count() if (n := f"{prefix}{i}") not in taken
            )
            taken.add(name)
            return name

        for of in fifos:
            if of.name is None:
                of.name = fresh("of")
        for b in rtps:
            if b._name is None:
                b._name = fresh("rtp")

    def _print_verify(self, ctx):
        verify = ctx.module.operation.verify()
        if not verify:
            raise RuntimeError(f"MLIR module failed verification: {verify}")
