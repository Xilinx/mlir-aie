# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from typing import Generator

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import WireBundle  # type: ignore
from ...dialects.aie import AIEDevice, tile, TileOp, get_target_model  # type: ignore
from ..resolvable import Resolvable
from .tile import Tile

import re


class Device(Resolvable):
    """
    A base class for representations of a device of a specific type.

    Note: this class is abstract because it does not implement Resolve
    """

    class __DeviceTile(Resolvable):
        """
        Interior class for tiles objects owned by a particular device.
        This is needed to ensure we don't generate more than one MLIR operation corresponding
        to the same logical tile within a device.
        """

        def __init__(self, col: int, row: int) -> None:
            self._col: int = col
            self._row: int = row
            self._op: TileOp | None = None
            super().__init__()

        def resolve(
            self,
            loc: ir.Location | None = None,
            ip: ir.InsertionPoint | None = None,
            allocation_scheme: str | None = None,
        ) -> None:
            if self._op == None:
                self._op = tile(
                    self._col,
                    self._row,
                    loc=loc,
                    ip=ip,
                    allocation_scheme=allocation_scheme,
                )

        @property
        def op(self) -> TileOp:
            if not self._op:
                raise ValueError("Cannot get operation before it is set.")
            return self._op

        @op.setter
        def op(self, op: TileOp):
            if self._op:
                raise ValueError("Cannot set operation more than once.")
            self._op = op

    def __init__(self, device: AIEDevice) -> None:
        """Initialize a representation of a device.

        Args:
            device (AIEDevice): aie device
        """
        self._device = device
        self._tiles: list[list[Device.__DeviceTile]] = []
        self._tm = get_target_model(device)
        for c in range(self._tm.columns()):
            self._tiles.append([])
            for r in range(self._tm.rows()):
                self._tiles[c].append(Device.__DeviceTile(c, r))

    def tile_iterator(self) -> Generator[Tile, None, None]:
        """
        Iterates over the device tiles deterministically
        """
        for c in range(self._tm.columns()):
            for r in range(self._tm.rows()):
                yield self._tiles[c][r]
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
            Tile(t._col, t._row)
            for t in self.tile_iterator()
            if self._tm.is_shim_noc_or_pl_tile(t._col, t._row)
        ]

    def get_mem_tiles(self) -> list[Tile]:
        """Returns a list of all mem tiles on the device.

        Returns:
            list[Tile]: A list of mem tiles.
        """
        return [
            Tile(t._col, t._row)
            for t in self.tile_iterator()
            if self._tm.is_mem_tile(t._col, t._row)
        ]

    def get_compute_tiles(self) -> list[Tile]:
        """Returns a list of all compute tiles on the device.

        Returns:
            list[Tile]: A list of compute tiles.
        """
        return [
            Tile(t._col, t._row)
            for t in self.tile_iterator()
            if self._tm.is_core_tile(t._col, t._row)
        ]

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

    def is_mem_accessible(self, source_tile: Tile, tiles: list[Tile]) -> bool:
        """Returns whether there exists a memory region on source_tile which all destination tiles can access.
        Returns:
            int: Number of connections (channels) available on the tile
        """
        if not isinstance(source_tile, Tile):
            raise ValueError(f"Expected a source Tile, but got {t}")
        for t in tiles:
            if not isinstance(t, Tile):
                raise ValueError(f"Expected a Tile, but got {t}")
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
        self._tiles[tile.col][tile.row].resolve(loc, ip, tile.allocation_scheme)
        tile.op = self._tiles[tile.col][tile.row].op


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
