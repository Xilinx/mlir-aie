"""
TODOs:
* docs
* error handling
* tile types"
"""

from ...dialects.aie import TileOp


class Tile:
    def __init__(self, col: int, row: int) -> None:
        self.col: int = col
        self.row: int = row
        self._op: TileOp | None = None

    @property
    def op(self) -> TileOp:
        assert self._op != None
        return self._op

    @op.setter
    def op(self, op: TileOp):
        assert self._op == None or self._op == op
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
    pass


class AnyMemTile:
    pass


class AnyComputeTile:
    pass


PlacementTile = Tile | AnyShimTile | AnyMemTile | AnyComputeTile
