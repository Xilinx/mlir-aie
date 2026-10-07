# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import itertools
from collections.abc import Sequence
from contextlib import contextmanager

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import TraceMode, device
from ..dialects.aiex import ConfigureOp
from ..helpers.dialects.func import FuncBase
from ..utils import trace as trace_utils
from ..utils.compile.jit.context import get_compile_arg
from .dataflow.objectfifo import ObjectFifoLink
from .device import Device
from .kernel import Kernel
from .resolvable import PerDeviceConfiguration, Resolvable
from .runtime import Runtime
from .runtime._context import active_configuration, active_configuration_scope
from .scratchpad_parameter import ScratchpadParameter


class DeviceConfiguration:
    """A device image and the runtime sequences that use it."""

    def __init__(
        self,
        name: str,
        device: Device,
        *,
        workers: Sequence = (),
        runtimes: Sequence[Runtime] = (),
    ) -> None:
        if not name:
            raise ValueError("DeviceConfiguration name must not be empty.")
        self._name = name
        self._device = device
        self._workers = list(workers)
        self._runtimes = list(runtimes)
        self._flows = []
        self._locks = []
        self._tile_dmas = []
        self._resolved_tile_dmas = None
        self._trace_size = None
        self._trace_workers = None
        self._reuse_output_buffer = False
        self._egress_shim_col = 0
        self._coretile_events = None
        self._coremem_events = None
        self._memtile_events = None
        self._shimtile_events = None
        self._core_trace_mode = TraceMode.EventTime
        self._owners = None

        runtime_names = [runtime.name for runtime in self._runtimes]
        duplicates = sorted(
            name for name in set(runtime_names) if runtime_names.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                f"DeviceConfiguration {self._name!r} has duplicate runtime names: {duplicates}."
            )
        for runtime in self._runtimes:
            runtime._bind_configuration(self)

    def _claim(self, resource, owners: dict[int, "DeviceConfiguration"]) -> None:
        if isinstance(resource, (Kernel, ScratchpadParameter)):
            return
        if isinstance(resource, PerDeviceConfiguration):
            resource._bind_device_configuration(self)
            return
        owner = owners.get(id(resource))
        if owner is not None and owner is not self:
            raise ValueError(
                f"{type(resource).__name__} already belongs to device configuration "
                f"{owner.name!r}; it cannot also belong to {self.name!r}."
            )
        owners[id(resource)] = self

    def claim_runtime_resource(self, resource) -> None:
        if self._owners is None:
            raise RuntimeError(
                "DeviceConfiguration cannot claim runtime resources before resolution."
            )
        self._claim(resource, self._owners)
        self._claim(resource.tile, self._owners)

    def _claim_fifo_graph(self, handle, owners, visited=None) -> None:
        if visited is None:
            visited = set()
        fifo = handle._object_fifo
        if id(fifo) in visited:
            return
        visited.add(id(fifo))
        self._claim(fifo, owners)
        runtime_handles = {
            runtime_handle
            for runtime in self._runtimes
            for runtime_handle in runtime.fifos
        }
        handles = [candidate for candidate in [fifo._prod, *fifo._cons] if candidate]
        for candidate in handles:
            self._claim(candidate, owners)
            endpoint = candidate.endpoint
            if endpoint in self._workers:
                pass
            elif candidate in runtime_handles:
                pass
            elif not isinstance(endpoint, ObjectFifoLink):
                raise ValueError(
                    f"ObjectFifo {fifo.name!r} has an endpoint outside device "
                    f"configuration {self.name!r}."
                )
            if isinstance(endpoint, PerDeviceConfiguration):
                self._claim(endpoint, owners)
            if isinstance(endpoint, ObjectFifoLink):
                self._claim(endpoint, owners)
                for linked in [*endpoint._srcs, *endpoint._dsts]:
                    if linked._object_fifo is not fifo:
                        self._claim_fifo_graph(linked, owners, visited)

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> Device:
        return self._device

    @property
    def workers(self) -> list:
        return list(self._workers)

    @property
    def runtimes(self) -> list[Runtime]:
        return list(self._runtimes)

    @contextmanager
    def configure(self):
        if active_configuration() is not None:
            raise RuntimeError("Nested configuration scopes are not supported.")
        op = ConfigureOp(self._name)
        block = ir.Block.create_at_start(op.body)
        with ir.InsertionPoint(block), active_configuration_scope(self):
            yield self

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
    ) -> None:
        self._trace_size = trace_size
        self._trace_workers = workers
        self._reuse_output_buffer = reuse_output_buffer
        self._coretile_events = coretile_events
        self._coremem_events = coremem_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events
        self._core_trace_mode = core_trace_mode
        self._egress_shim_col = egress_shim_col

    def add_flow(self, flow) -> None:
        if flow not in self._flows:
            self._flows.append(flow)

    def add_lock(self, lock) -> None:
        if lock not in self._locks:
            self._locks.append(lock)

    def add_tile_dma(self, tile_dma) -> None:
        from .runtime.runtime import IronRuntimeError

        if self._resolved_tile_dmas is not None:
            raise IronRuntimeError("Cannot register TileDma after DMA resolution.")
        self._tile_dmas.append(tile_dma)

    @property
    def flows(self) -> list:
        return list(self._flows)

    @property
    def locks(self) -> list:
        return list(self._locks)

    @property
    def tile_dmas(self) -> list:
        return list(self._tile_dmas)

    def resolve_tile_dmas(self) -> None:
        from .dataflow.tile_dma import TileDma
        from .runtime.runtime import IronRuntimeError

        if self._resolved_tile_dmas is None:
            programs = {}
            coordinates = {}
            for tile_dma in self._tile_dmas:
                tile = tile_dma.tile
                if tile.col is not None and tile.row is not None:
                    key = (tile.col, tile.row)
                    if key in coordinates and coordinates[key] is not tile:
                        raise IronRuntimeError(
                            f"Two TileDma programs name {tile}, via different "
                            "Tile objects. Share one Tile object for their channels."
                        )
                    coordinates[key] = tile
                if tile in programs:
                    programs[tile] = TileDma(
                        tile, [*programs[tile].channels, *tile_dma.channels]
                    )
                else:
                    programs[tile] = tile_dma
            self._resolved_tile_dmas = list(programs.values())
        for program in self._resolved_tile_dmas:
            program.resolve()

    def _name_unnamed(self) -> None:
        fifos = [
            h._object_fifo for runtime in self._runtimes for h in runtime.fifos
        ]
        fifos += [h._object_fifo for worker in self._workers for h in worker.fifos]
        fifos = list(dict.fromkeys(fifos))
        for fifo in fifos:
            for handle in [fifo._prod, *fifo._cons]:
                link = handle.endpoint if handle is not None else None
                if isinstance(link, ObjectFifoLink):
                    reached = [h._object_fifo for h in [*link._srcs, *link._dsts]]
                    fifos += [
                        reached_fifo
                        for reached_fifo in reached
                        if reached_fifo not in fifos
                    ]
        rtps = [
            buffer
            for worker in self._workers
            for buffer in worker.buffers
            if buffer._use_write_rtp
        ]
        taken = {fifo.name for fifo in fifos} | {buffer._name for buffer in rtps}

        def fresh(prefix):
            name = next(
                candidate
                for index in itertools.count()
                if (candidate := f"{prefix}{index}") not in taken
            )
            taken.add(name)
            return name

        for fifo in fifos:
            if fifo.name is None:
                fifo.name = fresh("of")
        for buffer in rtps:
            if buffer._name is None:
                buffer._name = fresh("rtp")

    def resolve(
        self,
        *,
        device_name: str,
        loc: ir.Location,
        entry: Runtime,
        owners: dict[int, "DeviceConfiguration"],
    ) -> None:
        """Emit this configuration as one ``aie.device`` operation."""
        self._owners = owners
        self._name_unnamed()
        device_type = type(self._device)
        self._device = device_type()  # pyright: ignore[reportCallIssue]
        current_device = self._device

        for runtime in self._runtimes:
            self._claim(runtime, owners)
        for worker in self._workers:
            self._claim(worker, owners)

        @device(current_device.resolve(), sym_name=device_name, loc=loc)
        def device_body():
            all_fifos = set()
            for runtime in self._runtimes:
                all_fifos.update(runtime.fifos)
            for worker in self._workers:
                all_fifos.update(worker.fifos)
            all_fifos = sorted(all_fifos, key=lambda obj: obj.name)

            all_tiles = []
            for worker in self._workers:
                all_tiles.append(worker.tile)
                for barrier in worker._barriers:
                    self._claim(barrier, owners)
                for arg in worker.flat_fn_args:
                    if isinstance(arg, Resolvable):
                        self._claim(arg, owners)
                        all_tiles.extend(arg.tiles())
            for handle in all_fifos:
                self._claim_fifo_graph(handle, owners)
                all_tiles.extend(
                    [endpoint.tile for endpoint in handle.all_of_endpoints()]
                )
                delegate = handle._object_fifo._delegate_tile
                if delegate is not None:
                    all_tiles.append(delegate)
            for flow in self._flows:
                self._claim(flow, owners)
                all_tiles.extend(flow.all_tiles())
            for tile_dma in self._tile_dmas:
                self._claim(tile_dma, owners)
                all_tiles.extend(tile_dma.all_tiles())
                buffers, locks = tile_dma.all_buffers_and_locks()
                for resource in [*buffers, *locks]:
                    self._claim(resource, owners)
            for lock in self._locks:
                self._claim(lock, owners)
                all_tiles.append(lock.tile)

            for tile in all_tiles:
                self._claim(tile, owners)
                current_device.resolve_tile(tile)

            for handle in all_fifos:
                handle.resolve()
            for lock in self._locks:
                lock.resolve()
            for tile_dma in self._tile_dmas:
                buffers, locks = tile_dma.all_buffers_and_locks()
                for lock in locks:
                    lock.resolve()
                for buffer in buffers:
                    buffer.place(tile_dma.tile)
                    buffer.resolve()

            for worker in self._workers:
                for arg in worker.flat_fn_args:
                    if isinstance(arg, FuncBase):
                        arg.emit()
                    elif isinstance(arg, Resolvable):
                        if arg not in self._flows and arg not in self._tile_dmas:
                            arg.resolve()

            for worker in self._workers:
                worker.resolve()
            for worker in self._workers:
                for cascade in worker._outgoing_cascades:
                    if cascade._dst not in self._workers:
                        raise ValueError(
                            "CascadeFlow endpoints must belong to the same device "
                            f"configuration {self.name!r}."
                        )
                    self._claim(cascade, owners)
                    self._claim(cascade._dst, owners)
                    cascade.resolve()

            self.resolve_tile_dmas()

            if self._trace_workers is not None:
                tiles_to_trace = [worker.tile.op for worker in self._trace_workers]
            else:
                tiles_to_trace = [
                    worker.tile.op
                    for worker in self._workers
                    if worker.trace is not None
                ]
            if self._trace_size is not None and self._trace_size > 0:
                trace_utils.configure_trace(
                    tiles_to_trace,
                    coretile_events=self._coretile_events,
                    coremem_events=self._coremem_events,
                    memtile_events=self._memtile_events,
                    shimtile_events=self._shimtile_events,
                    core_trace_mode=self._core_trace_mode,
                )

            for runtime in self._runtimes:
                load_pdi_device_ref = (
                    device_name
                    if runtime is entry and get_compile_arg("_iron_full_elf")
                    else None
                )
                runtime.resolve(
                    trace_size=self._trace_size,
                    reuse_output_buffer=self._reuse_output_buffer,
                    egress_shim_col=self._egress_shim_col,
                    load_pdi_device_ref=load_pdi_device_ref,
                    device=current_device,
                )

            for flow in self._flows:
                flow.resolve()
