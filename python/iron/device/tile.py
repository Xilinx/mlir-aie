# tile.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

from __future__ import annotations

from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
)
from ...dialects.aie import LogicalTileOp
from ...helpers.sourceloc import capture_source_site


class Tile:
    """An object representing a tile on a device.

    A tile can be:
    - Fully placed: col and row specified (e.g., Tile(0, 2))
    - Partially placed: only col or row (e.g., Tile(col=0))
    - Unplaced: no coordinates (e.g., Tile() or AnyComputeTile)

    The tile_type is normally set by the component that owns the tile
    (Worker sets CoreTile, ObjectFifoLink sets MemTile, RuntimeEndpoint
    sets ShimNOCTile). Users only need to specify tile_type for the rare
    case of overriding the default (e.g., forwarding through a compute
    tile instead of a mem tile).

    Pre-built Tile instances (AnyComputeTile, AnyMemTile, AnyShimTile)
    are provided as convenient defaults for tile-type-only requests.

    packet_type/packet_id set the tile's control-packet address
    (lowers to the `controller_id` attribute on the resolved `aie.tile`).
    """

    def __init__(
        self,
        col: int | None = None,
        row: int | None = None,
        *,
        tile_type: AIETileType | None = None,
        packet_type: int = 0,
        packet_id: int | None = None,
    ) -> None:
        self.col: int | None = col
        self.row: int | None = row
        self.tile_type: AIETileType | None = tile_type
        self.packet_type: int = packet_type
        self.packet_id: int | None = packet_id
        self._op: LogicalTileOp | None = None
        self._source_site = capture_source_site()

    def copy(self) -> Tile:
        """Return a copy of this Tile, including its control-packet id."""
        clone = Tile(
            self.col,
            self.row,
            tile_type=self.tile_type,
            packet_type=self.packet_type,
            packet_id=self.packet_id,
        )
        clone._source_site = self._source_site or capture_source_site()
        return clone

    def with_type(
        self,
        tile_type: AIETileType,
        *,
        mismatch_msg: str | None = None,
    ) -> Tile:
        """Return a fresh Tile with ``tile_type`` stamped, preserving col/row.

        Always returns a new object — never mutates self. Raises ValueError if
        the existing tile_type conflicts with tile_type.
        """
        if self.tile_type is not None and self.tile_type != tile_type:
            raise ValueError(
                mismatch_msg
                or f"Expected a {tile_type} tile, but got tile_type={self.tile_type}"
            )
        clone = Tile(
            self.col,
            self.row,
            tile_type=tile_type,
            packet_type=self.packet_type,
            packet_id=self.packet_id,
        )
        clone._source_site = self._source_site or capture_source_site()
        return clone

    @property
    def effective_tile_type(self) -> AIETileType | None:
        """Return the explicit type or the type inferred by Device.resolve_tile().

        DMA regions and shim routes need this when a Tile supplies only coordinates.
        Return None if neither an explicit type nor a resolved op is available.
        """
        if self.tile_type is not None:
            return self.tile_type
        if self._op is None:
            return None
        return AIETileType(int(self._op.tile_type))

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
        return id(self) == id(other)

    def __str__(self) -> str:
        col_str = str(self.col) if self.col is not None else "?"
        row_str = str(self.row) if self.row is not None else "?"
        type_str = f"<{self.tile_type}>" if self.tile_type is not None else ""
        return f"Tile{type_str}({col_str}, {row_str})"

    def __hash__(self):
        return id(self)


# Pre-built Tile instances specifying tile type
AnyShimTile = Tile(tile_type=AIETileType.ShimNOCTile)
AnyMemTile = Tile(tile_type=AIETileType.MemTile)
AnyComputeTile = Tile(tile_type=AIETileType.CoreTile)

# These are declared here, not by a user, so they carry no source site of their
# own; whoever adopts one (a Worker, an endpoint) supplies it instead. Without
# this they would otherwise report the user's `import aie.iron` line.
for _singleton in (AnyShimTile, AnyMemTile, AnyComputeTile):
    _singleton._source_site = None
del _singleton
