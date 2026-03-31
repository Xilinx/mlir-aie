# tile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ...dialects._aie_enum_gen import AIETileType
from ...dialects.aie import LogicalTileOp, TileOp


class TileType:
    """User-facing constants for AIE tile types."""

    Core = AIETileType.CoreTile
    Mem = AIETileType.MemTile
    ShimNOC = AIETileType.ShimNOCTile
    ShimPL = AIETileType.ShimPLTile


class Tile:
    """An object representing a tile on a device.

    A tile can be:
    - Fully placed: col and row specified (e.g., Tile(0, 2))
    - Partially placed: only col or row (e.g., Tile(col=0))
    - Unplaced: no coordinates (e.g., Tile())

    The tile_type is normally set by the component that owns the tile
    (Worker sets CoreTile, ObjectFifoLink sets MemTile, RuntimeEndpoint
    sets ShimNOCTile). Users only need to specify tile_type for the rare
    case of overriding the default (e.g., forwarding through a compute
    tile instead of a mem tile).
    """

    def __init__(
        self,
        col: int | None = None,
        row: int | None = None,
        *,
        tile_type: AIETileType | None = None,
        allocation_scheme: str | None = None,
    ) -> None:
        self.col: int | None = col
        self.row: int | None = row
        self.tile_type: AIETileType | None = tile_type
        self.allocation_scheme: str | None = allocation_scheme
        self._op: LogicalTileOp | TileOp | None = None

    @property
    def op(self) -> LogicalTileOp | TileOp:
        if not self._op:
            raise ValueError("Cannot get op before it is set.")
        return self._op

    @op.setter
    def op(self, op: LogicalTileOp | TileOp):
        if self._op and self._op != op:
            raise ValueError("Cannot change operation once it is set.")
        self._op = op

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return id(self) == id(other)

    def __str__(self) -> str:
        col_str = str(self.col) if self.col is not None else "?"
        row_str = str(self.row) if self.row is not None else "?"
        type_str = f"<{self.tile_type}>" if self.tile_type is not None else ""
        return f"Tile{type_str}({col_str}, {row_str})"

    def __hash__(self):
        return id(self)
