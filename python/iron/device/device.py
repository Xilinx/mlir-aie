# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from typing import Generator

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import AIETileType, WireBundle  # type: ignore
from ...dialects.aie import (
    AIEDevice,
    logical_tile,
    LogicalTileOp,
    get_target_model,
)  # type: ignore
from ..resolvable import Resolvable
from .tile import Tile

import re


class Device(Resolvable):
    """A representation of a device of a specific type.

    Provides device metadata (column/row counts) and emits
    aie.logical_tile ops for Tile objects during resolve.
    """

    def __init__(self, device: AIEDevice) -> None:
        self._device = device
        self._tm = get_target_model(device)
        self._resolved_tiles: dict[int, LogicalTileOp] = {}

    @property
    def cols(self) -> int:
        """Number of columns in the device tile array."""
        return self._tm.columns()

    @property
    def rows(self) -> int:
        """Number of rows in the device tile array."""
        return self._tm.rows()

    def _infer_tile_type(self, col, row):
        """Infer the AIETileType from device coordinates."""
        if self._tm.is_core_tile(col, row):
            return AIETileType.CoreTile
        elif self._tm.is_mem_tile(col, row):
            return AIETileType.MemTile
        elif self._tm.is_shim_noc_tile(col, row):
            return AIETileType.ShimNOCTile
        elif self._tm.is_shim_pl_tile(col, row):
            return AIETileType.ShimPLTile
        else:
            raise ValueError(f"Cannot determine tile type for ({col}, {row})")

    def tile_iterator(self) -> Generator[Tile, None, None]:
        """
        Iterates over the device tiles deterministically
        """
        for c in range(self._tm.columns()):
            for r in range(self._tm.rows()):
                yield Tile(c, r, tile_type=self._infer_tile_type(c, r))
        return None

    def get_shim_tiles(self) -> list[Tile]:
        """Returns a list of all shim tiles on the device.

        Returns:
            list[Tile]: A list of shim tiles.
        """
        return [
            Tile(t.col, t.row)
            for t in self.tile_iterator()
            if self._tm.is_shim_noc_or_pl_tile(t.col, t.row)
        ]

    def get_mem_tiles(self) -> list[Tile]:
        """Returns a list of all mem tiles on the device.

        Returns:
            list[Tile]: A list of mem tiles.
        """
        return [
            Tile(t.col, t.row)
            for t in self.tile_iterator()
            if self._tm.is_mem_tile(t.col, t.row)
        ]

    def get_compute_tiles(self) -> list[Tile]:
        """Returns a list of all compute tiles on the device.

        Returns:
            list[Tile]: A list of compute tiles.
        """
        return [
            Tile(t.col, t.row)
            for t in self.tile_iterator()
            if self._tm.is_core_tile(t.col, t.row)
        ]

    def get_num_source_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the switchbox for the given tile on the device.

        Args:
            t (Tile): The tile to query.

        Returns:
            int: Number of DMA source ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(f"Cannot query connections for unplaced tile: {t}")
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_source_switchbox_connections(col, row, bundle)

    def get_num_dest_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the switchbox for the given tile on the device.

        Args:
            t (Tile): The tile to query.

        Returns:
            int: Number of DMA dest ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(f"Cannot query connections for unplaced tile: {t}")
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_switchbox_connections(col, row, bundle)

    def get_num_source_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the shim mux for the given tile on the device.

        Args:
            t (Tile): The tile to query.

        Returns:
            int: Number of DMA source ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(f"Cannot query connections for unplaced tile: {t}")
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_source_shim_mux_connections(col, row, bundle)

    def get_num_dest_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the shim mux for the given tile on the device.

        Args:
            t (Tile): The tile to query.

        Returns:
            int: Number of DMA dest ports.
        """
        if t.col is None or t.row is None:
            raise ValueError(f"Cannot query connections for unplaced tile: {t}")
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_shim_mux_connections(col, row, bundle)

    def get_num_connections(self, tile: Tile, output: bool) -> int:
        """Returns number of DMA input or output "channels" available on the tile.
        Returns:
            int: Number of connections (channels) available on the tile.
        """
        if tile.col is None or tile.row is None:
            raise ValueError(f"Cannot query connections for unplaced tile: {tile}")
        if tile.row == 0:
            if output:
                return self.get_num_source_shim_mux_connections(tile)
            else:
                return self.get_num_dest_shim_mux_connections(tile)
        if output:
            return self.get_num_source_switchbox_connections(tile)
        else:
            return self.get_num_dest_switchbox_connections(tile)

    def is_mem_accessible(self, source_tile: Tile, tiles: list[Tile]) -> bool:
        """Returns whether there exists a memory region on source_tile which all destination tiles can access.
        Returns:
            bool: True if the given source tile has a memory region accessible by all destination tiles.
        """
        if not isinstance(source_tile, Tile):
            raise ValueError(f"Expected a source Tile, but got {source_tile}")
        for t in tiles:
            if not isinstance(t, Tile):
                raise ValueError(f"Expected a Tile, but got {t}")
        if source_tile.col is None or source_tile.row is None:
            raise ValueError(
                f"Cannot check memory accessibility for unplaced tile: {source_tile}"
            )
        for t in tiles:
            if t.col is None or t.row is None:
                raise ValueError(
                    f"Cannot check memory accessibility for unplaced tile: {t}"
                )
        if not tiles:
            return True

        source_is_compute = self._tm.is_core_tile(source_tile.col, source_tile.row)
        source_is_mem = self._tm.is_mem_tile(source_tile.col, source_tile.row)
        source_is_shim = self._tm.is_shim_noc_or_pl_tile(
            source_tile.col, source_tile.row
        )

        if source_is_compute and not all(
            [self._tm.is_core_tile(dst_tile.col, dst_tile.row) for dst_tile in tiles]
        ):
            return False
        if source_is_mem and not all(
            [self._tm.is_mem_tile(dst_tile.col, dst_tile.row) for dst_tile in tiles]
        ):
            return False
        if source_is_shim or any(
            [
                self._tm.is_shim_noc_or_pl_tile(dst_tile.col, dst_tile.row)
                for dst_tile in tiles
            ]
        ):
            # No neighbor sharing from shim tiles.
            return False

        for t in tiles:
            if not self._tm.is_legal_mem_affinity(
                source_tile.col, source_tile.row, t.col, t.row
            ):
                return False
        return True

    def resolve_tile(
        self,
        tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        tile_id = id(tile)
        if tile_id in self._resolved_tiles:
            tile.op = self._resolved_tiles[tile_id]
            return

        if tile.tile_type is None:
            if tile.col is not None and tile.row is not None:
                tile.tile_type = self._infer_tile_type(tile.col, tile.row)
            else:
                raise ValueError(
                    f"Cannot resolve {tile}: tile_type must be set or inferred from coordinates."
                )
        elif tile.col is not None and tile.row is not None:
            inferred = self._infer_tile_type(tile.col, tile.row)
            if tile.tile_type != inferred:
                raise ValueError(
                    f"Tile at ({tile.col}, {tile.row}) has tile_type={tile.tile_type}, "
                    f"but coordinates indicate {inferred}"
                )

        op = logical_tile(
            tile.tile_type,
            col=tile.col,
            row=tile.row,
            allocation_scheme=tile.allocation_scheme,
            loc=loc,
            ip=ip,
        )
        self._resolved_tiles[tile_id] = op
        tile.op = op


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
