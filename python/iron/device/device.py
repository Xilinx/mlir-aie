# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import abstractmethod
from typing import Generator

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import WireBundle  # type: ignore
from ...dialects.aie import AIEDevice, tile, TileOp, get_target_model  # type: ignore
from ..resolvable import Resolvable
from .tile import Tile

import re


class DeviceLike(Resolvable):
    class __DeviceTile(Tile, Resolvable):
        """
        Interior class for tiles objects owned by a particular device.
        This is needed to ensure we don't generate more than one MLIR operation corresponding
        to the same logical tile within a device.
        """

        def __init__(self, col: int, row: int) -> None:
            Tile.__init__(self, col, row)
            Resolvable.__init__(self)

        def resolve(
            self,
            loc: ir.Location | None = None,
            ip: ir.InsertionPoint | None = None,
            allocation_scheme: str | None = None,
        ) -> None:
            if self._op == None:
                self._op = tile(
                    self.col,
                    self.row,
                    loc=loc,
                    ip=ip,
                    allocation_scheme=allocation_scheme,
                )

    def __init__(
        self, device: AIEDevice, tiles: list[list[Tile]] | None = None
    ) -> None:
        self._device = device
        self._tm = get_target_model(device)
        if not tiles:
            self._tiles: list[list[DeviceLike.__DeviceTile]] = []
            for c in range(self._tm.columns()):
                self._tiles.append([])
                for r in range(self._tm.rows()):
                    self._tiles[c].append(DeviceLike.__DeviceTile(c, r))
        else:
            self._tiles = tiles

    def get_shim_tiles(self) -> list[Tile]:
        """Returns a list of all shim tiles on the device.

        Returns:
            list[Tile]: A list of shim tiles.
        """
        return [
            t
            for t in self.tile_iterator()
            if self._tm.is_shim_noc_or_pl_tile(t.col, t.row)
        ]

    def get_mem_tiles(self) -> list[Tile]:
        """Returns a list of all mem tiles on the device.

        Returns:
            list[Tile]: A list of mem tiles.
        """
        return [t for t in self.tile_iterator() if self._tm.is_mem_tile(t.col, t.row)]

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

    def get_compute_tiles(self) -> list[Tile]:
        """Returns a list of all compute tiles on the device.

        Returns:
            list[Tile]: A list of compute tiles.
        """
        return [t for t in self.tile_iterator() if self._tm.is_core_tile(t.col, t.row)]

    def get_num_source_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_source_switchbox_connections(col, row, bundle)

    def get_num_dest_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_switchbox_connections(col, row, bundle)

    def get_num_source_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_source_shim_mux_connections(col, row, bundle)

    def get_num_dest_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return self._tm.get_num_dest_shim_mux_connections(col, row, bundle)

    def get_num_connections(self, tile: Tile, output: bool) -> int:
        """Returns number of DMA input or output "channels" available on the tile.
        Returns:
            int: Number of connections (channels) available on the tile
        """
        if tile.row == 0:
            if output:
                return self.get_num_source_shim_mux_connections(tile)
            else:
                return self.get_num_dest_shim_mux_connections(tile)
        if output:
            return self.get_num_source_switchbox_connections(tile)
        else:
            return self.get_num_dest_switchbox_connections(tile)

    def is_legal_mem_affinity(self, src_tile: Tile, dst_tile: Tile) -> bool:
        """Returns whether memory on a destination can be accessed by a source.
        Returns:
            int: Number of connections (channels) available on the tile
        """
        return self._tm.is_legal_mem_affinity(
            src_tile.col, src_tile.row, dst_tile.col, dst_tile.row
        )

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return self._device

    def resolve_tile(
        self,
        tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._tiles[tile.col][tile.row].resolve(loc, ip, tile.allocation_scheme)
        tile.op = self._tiles[tile.col][tile.row].op

    def tile_iterator(self) -> Generator[Tile, None, None]:
        """
        Iterates over the available device tiles deterministically
        """
        for c in range(self._tm.columns()):
            for r in range(self._tm.rows()):
                yield self._tiles[c][r]


class DeviceView(DeviceLike):
    def __init__(self, device: "Device", tiles: list[Tile]):
        super().__init__(device=device._device, tiles=tiles)
        self._device_instance = device
        self._coords = set()
        for col in tiles:
            for t in col:
                self._coords.add((t.col, t.row))

    def tile_iterator(self) -> Generator[Tile, None, None]:
        # Keep ordering consistent from the device we sliced from.
        for c in range(len(self._tiles)):
            for r in range(len(self._tiles[0])):
                yield self._tiles[c][r]


class Device(DeviceLike):
    """
    A base class for representations of a device of a specific type.
    """

    def __init__(self, device: AIEDevice) -> None:
        """Initialize a representation of a device.

        Args:
            device (AIEDevice): aie device
        """
        super().__init__(device=device)
        self._claimed_tiles = set()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) > 2:
                raise IndexError("Only 2D slicing is supported for devices.")
            if len(key) == 2:
                col_slice, row_slice = key
            else:
                col_slice, row_slice = key[0], slice(None, None, None)
        elif isinstance(key, (int, slice)):
            col_slice, row_slice = key, slice(None, None, None)
        else:
            raise IndexError(
                "Device indices must be integers, slices, or a 2-tuple of those."
            )

        # Special case for single tile access - this is non-destructive
        # TODO(erika): this should be destructive
        if isinstance(col_slice, int) and isinstance(row_slice, int):
            if col_slice >= self._tm.columns() or row_slice >= self._tm.rows():
                raise IndexError("Tile index out of range.")
            return [[self._tiles[col_slice][row_slice]]]

        # Handle slices and integers for cols
        if isinstance(col_slice, int):
            cols = [col_slice]
        else:
            cols = range(self._tm.columns())[col_slice]

        # Handle slices and integers for rows
        if isinstance(row_slice, int):
            rows = [row_slice]
        else:
            rows = range(self._tm.rows())[row_slice]

        if not cols or not rows:
            return DeviceView(self, [])

        tiles_to_claim = []
        coords_to_claim = set()
        for c in cols:
            tiles_to_claim.append([])
            for r in rows:
                if (c, r) in self._claimed_tiles:
                    raise ValueError(f"Tile ({c}, {r}) has already been claimed.")
                coords_to_claim.add((c, r))
                tiles_to_claim[-1].append(self._tiles[c][r])

        self._claimed_tiles.update(coords_to_claim)
        return DeviceView(self, tiles_to_claim)

    def tile_iterator(self) -> Generator[Tile, None, None]:
        for t in super().tile_iterator():
            if (t.col, t.row) not in self._claimed_tiles:
                yield self._tiles[t.col][t.row]


def create_class(class_name, device):

    def _device__init__(self) -> None:
        super(globals()[class_name], self).__init__(device=device)

    globals()[class_name] = type(
        class_name,
        (Device,),
        {
            "__init__": _device__init__,
            "__doc__": f"A representation of a device that resolves to {device}",
        },
    )


for device in AIEDevice:
    class_name = re.sub(r"NPU(\d+)_(\d+)COL", r"NPU\1Col\2", device.name.upper())
    create_class(class_name, device)
