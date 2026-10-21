# resolvable.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Structural protocol for objects that lower to MLIR operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

if TYPE_CHECKING:
    from .configuration import DeviceConfiguration


# Structural typing via @runtime_checkable Protocol: any class with both
# .resolve() and .tiles() passes isinstance(x, Resolvable).  The two-method
# requirement is the safeguard against false positives from classes that
# happen to define an unrelated .resolve() (e.g. pathlib.Path).
@runtime_checkable
class Resolvable(Protocol):
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Resolve the current object into one or more MLIR operations.

        Should only be called within an MLIR context.

        Args:
            loc (ir.Location | None, optional): Location is used by MLIR object during construction in some cases. Defaults to None.
            ip (ir.InsertionPoint | None, optional): InsertionPoint is used by MLIR object during construction in some cases. Defaults to None.
        """
        ...

    def tiles(self) -> list:
        """Tiles this Resolvable depends on for code generation.

        Override this in user-side Resolvable subclasses that reference tiles
        which aren't already discoverable via Workers or ObjectFifos. The
        DeviceConfiguration resolves these tiles before calling `resolve`, so
        `tile.op` is valid by then. Default: empty list.
        """
        return []


class PerDeviceConfiguration:
    """An object that belongs to one ``aie.device`` image."""

    _device_configuration: DeviceConfiguration | None = None

    def _bind_device_configuration(self, configuration: DeviceConfiguration) -> None:
        owner = self._device_configuration
        if owner is not None and owner is not configuration:
            raise ValueError(
                f"{type(self).__name__} already belongs to device configuration "
                f"{owner.name!r}; it cannot also belong to {configuration.name!r}."
            )
        self._device_configuration = configuration


class PerDeviceConfigurationResolvable(PerDeviceConfiguration, Resolvable):
    """A resolvable whose emitted operations belong to one ``aie.device``.

    User-defined worker arguments that emit device-local operations should
    inherit this class. Structural ``Resolvable`` implementations remain
    supported, but this base reports cross-configuration reuse at the object.
    """


class DeviceResources:
    """Flows, locks, and tile DMA programs owned by one device image."""

    def __init__(self) -> None:
        self._flows = []
        self._locks = []
        self._tile_dmas = []
        self._resolved_tile_dmas = None

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

    @property
    def flows(self) -> list:
        return list(self._flows)

    @property
    def locks(self) -> list:
        return list(self._locks)

    @property
    def tile_dmas(self) -> list:
        return list(self._tile_dmas)


class NotResolvedError(Exception):
    """Raised when a property or operation is accessed on a `Resolvable` object before `resolve` has been called."""

    def __init__(self, message="Cannot get operation; class not resolved."):
        self.message = message
        super().__init__(self.message)
