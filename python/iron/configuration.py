# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects.aie import TraceMode
from ..dialects.aiex import ConfigureOp
from .device import Device
from .runtime import Runtime
from .runtime._context import active_configuration, active_configuration_scope


class Configuration:
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
            raise ValueError("Configuration name must not be empty.")
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

        runtime_names = [runtime.name for runtime in self._runtimes]
        duplicates = sorted(
            name for name in set(runtime_names) if runtime_names.count(name) > 1
        )
        if duplicates:
            raise ValueError(
                f"Configuration {self._name!r} has duplicate runtime names: {duplicates}."
            )
        for runtime in self._runtimes:
            runtime._bind_configuration(self)

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
        if self._resolved_tile_dmas is not None:
            raise RuntimeError("Cannot register TileDma after DMA resolution.")
        if tile_dma not in self._tile_dmas:
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

        if self._resolved_tile_dmas is None:
            programs = {}
            coordinates = {}
            for tile_dma in self._tile_dmas:
                tile = tile_dma.tile
                if tile.col is not None and tile.row is not None:
                    key = (tile.col, tile.row)
                    if key in coordinates and coordinates[key] is not tile:
                        raise RuntimeError(
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