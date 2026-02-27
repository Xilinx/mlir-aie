# tile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.

from ...dialects.aie import LogicalTileOp


class Tile:
    """An object representing a component to be placed on a device, optionally denoted by coordinates."""

    # Tile type constants
    COMPUTE = "compute"  # Compute tiles
    MEMORY = "memory"  # Memory tiles
    SHIM = "shim"  # Shim tiles

    def __init__(
        self,
        col: int | None = None,
        row: int | None = None,
        *,
        tile_type: str | None = None,
        allocation_scheme: str | None = None,
    ) -> None:
        """Initialize a Tile with full, partial, or no coordinates.

        Args:
            col: Column coordinate (None for unconstrained)
            row: Row coordinate (None for unconstrained)
            tile_type: Tile type - use Tile.COMPUTE, Tile.MEMORY, or Tile.SHIM.
                       Can be inferred from context or coordinates.
            allocation_scheme: Optional allocation scheme string

        Examples:
            Tile(2, 3)                           # Full coords, type inferred
            Tile(col=2)                          # Partial, type from context
            Tile()                               # Unconstrained, type from context
            Tile(2, 3, tile_type=Tile.COMPUTE)   # Explicit validation
        """
        self.col: int | None = col
        self.row: int | None = row
        self.tile_type: str | None = tile_type
        self.allocation_scheme: str | None = allocation_scheme
        self._op: LogicalTileOp | None = None

        # Validate tile_type if specified
        if tile_type is not None:
            valid_types = [Tile.COMPUTE, Tile.MEMORY, Tile.SHIM]
            if tile_type not in valid_types:
                raise ValueError(
                    f"Invalid tile_type '{tile_type}'. Must be one of: "
                    f"Tile.COMPUTE, Tile.MEMORY, Tile.SHIM"
                )

    @property
    def op(self) -> LogicalTileOp:
        if not self._op:
            raise ValueError("Cannot get op before it is set.")
        return self._op

    @op.setter
    def op(self, op: LogicalTileOp):
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
