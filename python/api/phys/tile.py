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
        self.__op: TileOp | None = None

    @property
    def op(self) -> TileOp:
        assert self.__op != None
        return self.__op

    @op.setter
    def op(self, op: TileOp):
        assert self.__op == None
        self.__op = op
