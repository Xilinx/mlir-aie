# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import abstractmethod
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

        # Create all "physical" tiles belonging to the device at initialization to
        # ensure only one "physical" tile object is ever created corresponding to the same
        # coordinates.
        tm = get_target_model(device)
        for c in range(tm.columns()):
            self._tiles.append([])
            for r in range(tm.rows()):
                self._tiles[c].append(Device.__DeviceTile(c, r))

    @property
    def rows(self) -> int:
        return get_target_model(self._device).rows()

    @property
    def cols(self) -> int:
        return get_target_model(self._device).columns()

    @abstractmethod
    def get_shim_tiles(self) -> list[Tile]:
        """Returns a list of all shim tiles on the device.

        Returns:
            list[Tile]: A list of shim tiles.
        """
        ...

    @abstractmethod
    def get_mem_tiles(self) -> list[Tile]:
        """Returns a list of all mem tiles on the device.

        Returns:
            list[Tile]: A list of mem tiles.
        """
        ...

    @abstractmethod
    def get_compute_tiles(self) -> list[Tile]:
        """Returns a list of all compute tiles on the device.

        Returns:
            list[Tile]: A list of compute tiles.
        """
        # TODO: should this be shaped?
        ...

    @abstractmethod
    def get_num_source_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        ...

    @abstractmethod
    def get_num_dest_switchbox_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the switchbox for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        ...

    @abstractmethod
    def get_num_source_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA source ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA source ports.
        """
        ...

    @abstractmethod
    def get_num_dest_shim_mux_connections(self, t: Tile) -> int:
        """Returns number of DMA dest ports in the shim mux for the given tile on the device.

        Returns:
            int: Number of DMA dest ports.
        """
        ...

    def resolve_tile(
        self,
        tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._tiles[tile.col][tile.row].resolve(loc, ip, tile.allocation_scheme)
        tile.op = self._tiles[tile.col][tile.row].op


class NPUBase(Device):
    """A base class which can be used to create other device specific classes.
    This class is abstract because it does not implement resolve()

    This class makes some assumptions:
    * The 0th tile in each column is a shim tile
    * The 1st tile in each column is a mem tile
    * The 2nd+ tiles in each column are compute tiles
    """

    def __init__(self, device: AIEDevice) -> None:
        """Initialize a device based on the AIEDevice.

        Args:
            device (AIEDevice): aie device
        """
        super().__init__(device=device)

    def get_shim_tiles(self) -> list[Tile]:
        shim_tiles = []
        for col in range(self.cols):
            shim_tiles.append(Tile(col, 0))
        return shim_tiles

    def get_mem_tiles(self) -> list[Tile]:
        mem_tiles = []
        for col in range(self.cols):
            mem_tiles.append(Tile(col, 1))
        return mem_tiles

    def get_compute_tiles(self) -> list[Tile]:
        compute_tiles = []
        mem_tile_rows = get_target_model(self._device).get_num_mem_tile_rows()
        for col in range(self.cols):
            for row in range(1 + mem_tile_rows, self.rows):
                compute_tiles.append(Tile(col, row))
        return compute_tiles

    def get_num_source_switchbox_connections(self, t: Tile) -> int:
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return get_target_model(self._device).get_num_source_switchbox_connections(
            col, row, bundle
        )

    def get_num_dest_switchbox_connections(self, t: Tile) -> int:
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return get_target_model(self._device).get_num_dest_switchbox_connections(
            col, row, bundle
        )

    def get_num_source_shim_mux_connections(self, t: Tile) -> int:
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return get_target_model(self._device).get_num_source_shim_mux_connections(
            col, row, bundle
        )

    def get_num_dest_shim_mux_connections(self, t: Tile) -> int:
        col = t.col
        row = t.row
        bundle = WireBundle.DMA
        return get_target_model(self._device).get_num_dest_shim_mux_connections(
            col, row, bundle
        )

    def get_num_connections(self, tile: Tile, output: bool) -> int:
        if tile.row == 0:
            if output:
                return self.get_num_source_shim_mux_connections(tile)
            else:
                return self.get_num_dest_shim_mux_connections(tile)
        if output:
            return self.get_num_source_switchbox_connections(tile)
        else:
            return self.get_num_dest_switchbox_connections(tile)


def create_class(class_name, device):

    def _device__init__(self) -> None:
        super(globals()[class_name], self).__init__(device=device)

    def _device_resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return device

    base = NPUBase if "NPU" in class_name else Device
    globals()[class_name] = type(
        class_name,
        (base,),
        {
            "__init__": _device__init__,
            "resolve": _device_resolve,
            "__doc__": f"A representation of a device that resolves to {device}",
        },
    )


for device in AIEDevice:
    class_name = re.sub(r"NPU(\d+)_(\d+)COL", r"NPU\1Col\2", device.name.upper())
    create_class(class_name, device)
