# tile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ...dialects.aie import TileOp


class Tile:
    """An object representing a single component denoted by coordinates on a device."""

    def __init__(self, col: int, row: int, allocation_scheme: str = None) -> None:
        self.col: int = col
        self.row: int = row
        self.allocation_scheme: str | None = allocation_scheme
        self._op: TileOp | None = None
        # TODO: each tile should probably have a type, e.g., Shim or Mem or Compute

    @property
    def op(self) -> TileOp:
        if not self._op:
            raise ValueError("Cannot get op before it is set.")
        return self._op

    @op.setter
    def op(self, op: TileOp):
        if self._op and self._op != op:
            raise ValueError("Cannot change operation once it is set.")
        self._op = op

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return self.col == other.col and self.row == other.row

    def __str__(self) -> str:
        return f"Tile({self.col}, {self.row})"

    def __hash__(self):
        return hash(str(self))


class AnyShimTile:
    """A placeholder that should be replaced with a concrete Tile() representing a Shim tile on a device."""

    pass


class AnyMemTile:
    """A placeholder that should be replaced with a concrete Tile() representing a Mem tile on a device."""

    pass


class AnyComputeTile:
    """A placeholder that should be replaced with a concrete Tile() representing a Compute tile on a device."""

    pass


PlacementTile = Tile | AnyShimTile | AnyMemTile | AnyComputeTile
