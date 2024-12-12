# device.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import abstractmethod
from ... import ir  # type: ignore
from ...dialects.aie import AIEDevice, tile, TileOp  # type: ignore
from ..resolvable import Resolvable
from .tile import Tile

# TODO: we need an NPU2 implementation.


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
        ) -> None:
            if self._op == None:
                self._op = tile(self._col, self._row, loc=loc, ip=ip)

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

    def __init__(self, cols: int, rows: int) -> None:
        """Initialize a representation of a device.

        Args:
            cols (int): Number of columns on the device
            rows (int): Number of rows on the device.
        """
        self._cols = cols
        self._rows = rows
        self._tiles: list[list[Device.__DeviceTile]] = []

        # Create all "physical" tiles belonging to the device at initialization to
        # ensure only one "physical" tile object is every created corresponding to the same
        # coordinates.
        for c in range(self._cols):
            self._tiles.append([])
            for r in range(self._rows):
                self._tiles[c].append(Device.__DeviceTile(c, r))

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

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

    def resolve_tile(
        self,
        tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._tiles[tile.col][tile.row].resolve(loc, ip)
        tile.op = self._tiles[tile.col][tile.row].op


class NPU1Col1(Device):
    """A representation of a device that resolves to AIEDevice.npu1_1col"""

    def __init__(self) -> None:
        super().__init__(cols=1, rows=6)

    def get_shim_tiles(self) -> list[Tile]:
        return [Tile(0, 0)]

    def get_mem_tiles(self) -> list[Tile]:
        return [Tile(0, 1)]

    def get_compute_tiles(self) -> list[Tile]:
        return [Tile(0, row) for row in range(2, self._rows)]

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return AIEDevice.npu1_1col


class NPU1Col2(Device):
    """A representation of a device that resolves to AIEDevice.npu1_2col"""

    def __init__(self) -> None:
        super().__init__(cols=2, rows=6)

    def get_shim_tiles(self) -> list[Tile]:
        shim_tiles = []
        for col in range(self._cols):
            shim_tiles.append(Tile(col, 0))
        return shim_tiles

    def get_mem_tiles(self) -> list[Tile]:
        mem_tiles = []
        for col in range(self._cols):
            mem_tiles.append(Tile(col, 1))
        return mem_tiles

    def get_compute_tiles(self) -> list[Tile]:
        compute_tiles = []
        for col in range(self._cols):
            for row in range(2, self._rows):
                compute_tiles.append(Tile(col, row))
        return compute_tiles

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return AIEDevice.npu1_2col


class NPU1Col3(Device):
    """A representation of a device that resolves to AIEDevice.npu1_3col"""

    def __init__(self) -> None:
        super().__init__(cols=3, rows=6)

    def get_shim_tiles(self) -> list[Tile]:
        shim_tiles = []
        for col in range(self._cols):
            shim_tiles.append(Tile(col, 0))
        return shim_tiles

    def get_mem_tiles(self) -> list[Tile]:
        mem_tiles = []
        for col in range(self._cols):
            mem_tiles.append(Tile(col, 1))
        return mem_tiles

    def get_compute_tiles(self) -> list[Tile]:
        compute_tiles = []
        for col in range(self._cols):
            for row in range(2, self._rows):
                compute_tiles.append(Tile(col, row))
        return compute_tiles

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return AIEDevice.npu1_3col


class NPU1Col4(Device):
    """A representation of a device that resolves to AIEDevice.npu1_4col"""

    def __init__(self) -> None:
        super().__init__(cols=4, rows=6)

    def get_shim_tiles(self) -> list[Tile]:
        shim_tiles = []
        for col in range(self._cols):
            shim_tiles.append(Tile(col, 0))
        return shim_tiles

    def get_mem_tiles(self) -> list[Tile]:
        mem_tiles = []
        for col in range(self._cols):
            mem_tiles.append(Tile(col, 1))
        return mem_tiles

    def get_compute_tiles(self) -> list[Tile]:
        compute_tiles = []
        for col in range(self._cols):
            for row in range(2, self._rows):
                compute_tiles.append(Tile(col, row))
        return compute_tiles

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return AIEDevice.npu1_4col


class XCVC1902(Device):
    """A placeholder representation of a device that resolves to IEDevice.xcvc1902
    TODO: this needs to be implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError("This device type is not yet implementated")

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return AIEDevice.xcvc1902
