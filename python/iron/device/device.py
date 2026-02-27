# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
import re
from typing import Generator

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import WireBundle  # type: ignore
from ...dialects.aie import (  # type: ignore
    AIEDevice,
    AIETileType,
    logical_tile,
    LogicalTileOp,
    get_target_model,
)
from ..resolvable import Resolvable
from .tile import Tile


class Device(Resolvable):
    """A base class for representations of a device of a specific type."""

    def __init__(self, device: AIEDevice) -> None:
        """Initialize a representation of a device.

        Args:
            device (AIEDevice): aie device
        """
        self._device = device
        self._tm = get_target_model(device)
        self._allocated_compute_tiles: set[tuple[int, int]] = set()

    def _get_tile_type_from_coords(self, col: int, row: int) -> str:
        """Query device target model for IRON tile type at coordinates.

        Returns:
            Tile type string (Tile.COMPUTE, Tile.MEMORY, or Tile.SHIM)
        """
        if self._tm.is_core_tile(col, row):
            return Tile.COMPUTE
        elif self._tm.is_mem_tile(col, row):
            return Tile.MEMORY
        elif self._tm.is_shim_noc_tile(col, row):
            return Tile.SHIM
        elif self._tm.is_shim_pl_tile(col, row):
            return Tile.SHIM
        else:
            raise ValueError(f"Unknown tile type for coordinates ({col}, {row})")

    def tile_iterator(self) -> Generator[Tile, None, None]:
        """Iterates over the device tiles deterministically."""
        for c in range(self._tm.columns()):
            for r in range(self._tm.rows()):
                yield Tile(c, r)
        return None

    @property
    def rows(self) -> int:
        return self._tm.rows()

    @property
    def cols(self) -> int:
        return self._tm.columns()

    def get_shim_tiles(self) -> list[Tile]:
        """Returns a list of all shim tiles on the device.

        Returns:
            list[Tile]: A list of shim tiles.
        """
        return [
            Tile(c, r)
            for c in range(self._tm.columns())
            for r in range(self._tm.rows())
            if self._tm.is_shim_noc_or_pl_tile(c, r)
        ]

    def get_mem_tiles(self) -> list[Tile]:
        """Returns a list of all mem tiles on the device.

        Returns:
            list[Tile]: A list of mem tiles.
        """
        return [
            Tile(c, r)
            for c in range(self._tm.columns())
            for r in range(self._tm.rows())
            if self._tm.is_mem_tile(c, r)
        ]

    def get_compute_tiles(self) -> list[Tile]:
        """Returns a list of all compute tiles on the device.

        Returns:
            list[Tile]: A list of compute tiles.
        """
        return [
            Tile(c, r)
            for c in range(self._tm.columns())
            for r in range(self._tm.rows())
            if self._tm.is_core_tile(c, r)
        ]

    def get_num_source_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(
                f"get_num_source_switchbox_connections requires Tile with concrete coordinates, "
                f"got Tile(col={t.col}, row={t.row})"
            )
        bundle = WireBundle.DMA
        return self._tm.get_num_source_switchbox_connections(t.col, t.row, bundle)

    def get_num_dest_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(
                f"get_num_dest_switchbox_connections requires Tile with concrete coordinates, "
                f"got Tile(col={t.col}, row={t.row})"
            )
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_switchbox_connections(t.col, t.row, bundle)

    def get_num_source_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(
                f"get_num_source_shim_mux_connections requires Tile with concrete coordinates, "
                f"got Tile(col={t.col}, row={t.row})"
            )
        bundle = WireBundle.DMA
        return self._tm.get_num_source_shim_mux_connections(t.col, t.row, bundle)

    def get_num_dest_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(
                f"get_num_dest_shim_mux_connections requires Tile with concrete coordinates, "
                f"got Tile(col={t.col}, row={t.row})"
            )
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_shim_mux_connections(t.col, t.row, bundle)

    def get_num_connections(self, tile: Tile, output: bool) -> int:
        """Returns number of DMA input or output "channels" available on the tile.

        Returns:
            int: Number of connections (channels) available on the tile
        """
        if tile.col is None or tile.row is None:
            raise ValueError(
                f"get_num_connections requires Tile with concrete coordinates, "
                f"got Tile(col={tile.col}, row={tile.row})"
            )
        if tile.row == 0:
            if output:
                return self.get_num_source_shim_mux_connections(tile)
            else:
                return self.get_num_dest_shim_mux_connections(tile)
        if output:
            return self.get_num_source_switchbox_connections(tile)
        else:
            return self.get_num_dest_switchbox_connections(tile)

    def resolve_tile(
        self,
        placement_tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> LogicalTileOp:
        """Resolve a Tile to a LogicalTileOp.

        Tile can be fully constrained, partially constrained, or unconstrained.
        Tile type is inferred from coordinates if not explicitly specified.

        If the tile already has a LogicalTileOp stored, it is returned directly
        without creating a new one. This ensures tile reuse.

        Args:
            placement_tile: A Tile object (possibly with partial/no coordinates)
            loc: MLIR location
            ip: Insertion point

        Returns:
            LogicalTileOp: The created (or existing) logical tile operation
        """
        if not isinstance(placement_tile, Tile):
            raise ValueError(
                f"resolve_tile expects Tile object, got {type(placement_tile)}"
            )

        # If tile already has an op, return it (tile reuse)
        if placement_tile._op is not None:
            return placement_tile._op

        iron_to_mlir = {
            Tile.COMPUTE: AIETileType.CoreTile,
            Tile.MEMORY: AIETileType.MemTile,
            Tile.SHIM: AIETileType.ShimNOCTile,
        }

        # Determine IRON tile type
        iron_tile_type = placement_tile.tile_type

        # Infer tile type if not set
        if iron_tile_type is None:
            if placement_tile.col is not None and placement_tile.row is not None:
                iron_tile_type = self._get_tile_type_from_coords(
                    placement_tile.col, placement_tile.row
                )
            else:
                # tile_type should have been set by Worker/RuntimeEndpoint/ObjectFifoLink
                raise ValueError(
                    f"Tile type not set by context and cannot be inferred from coordinates. "
                    f"Tile(col={placement_tile.col}, row={placement_tile.row}) must have "
                    f"tile_type set by the context (Worker, RuntimeEndpoint, ObjectFifoLink)."
                )

        # If user specified both tile_type AND coordinates, check they match
        if (
            placement_tile.tile_type is not None
            and placement_tile.col is not None
            and placement_tile.row is not None
        ):
            actual_iron_type = self._get_tile_type_from_coords(
                placement_tile.col, placement_tile.row
            )

            if placement_tile.tile_type != actual_iron_type:
                raise ValueError(
                    f"Tile type mismatch: specified tile_type='{placement_tile.tile_type}' "
                    f"but coordinates ({placement_tile.col}, {placement_tile.row}) "
                    f"correspond to '{actual_iron_type}' on this device"
                )

        # Compute tiles cannot be shared, check for duplicates
        if iron_tile_type == Tile.COMPUTE:
            if placement_tile.col is not None and placement_tile.row is not None:
                coord_tuple = (placement_tile.col, placement_tile.row)
                if coord_tuple in self._allocated_compute_tiles:
                    raise ValueError(
                        f"Compute tile ({placement_tile.col}, {placement_tile.row}) "
                        f"already allocated. Each Worker requires a unique compute tile. "
                        f"Compute tiles cannot be shared between Workers."
                    )
                self._allocated_compute_tiles.add(coord_tuple)

        mlir_tile_type = iron_to_mlir[iron_tile_type]

        logical_tile_op = logical_tile(
            mlir_tile_type,
            col=placement_tile.col,
            row=placement_tile.row,
            loc=loc,
            ip=ip,
            allocation_scheme=placement_tile.allocation_scheme,
        )

        placement_tile._op = logical_tile_op
        return logical_tile_op


def create_class(class_name, device):

    def _device__init__(self) -> None:
        super(globals()[class_name], self).__init__(device=device)

    def _device_resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return device

    globals()[class_name] = type(
        class_name,
        (Device,),
        {
            "__init__": _device__init__,
            "resolve": _device_resolve,
            "__doc__": f"A representation of a device that resolves to {device}",
        },
    )


for device in AIEDevice:
    class_name = re.sub(r"NPU(\d+)_(\d+)COL", r"NPU\1Col\2", device.name.upper())
    create_class(class_name, device)
