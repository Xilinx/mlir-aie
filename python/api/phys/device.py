"""
TODOs:
* Tile types
* doc
* other devices
* error handling

devs = {
    1: AIEDevice.npu1_1col,
    2: AIEDevice.npu1_2col,
    3: AIEDevice.npu1_3col,
    4: AIEDevice.npu1_4col,
    5: AIEDevice.npu1,
}
"""

from ... import ir
from ...dialects.aie import AIEDevice, tile, TileOp
from ..resolvable import Resolvable
from .tile import MyTile


class MyDevice(Resolvable):
    """
    Note: this class is abstract because it does not implement Resolve
    """

    class __MyDeviceTile(Resolvable):
        """
        Interior class for tiles objects owned by a particular device
        """

        def __init__(self, col: int, row: int) -> None:
            self.__col: int = col
            self.__row: int = row
            self.__op: TileOp | None = None
            super().__init__()

        def resolve(
            self,
            loc: ir.Location = None,
            ip: ir.InsertionPoint = None,
        ) -> None:
            if self.__op == None:
                self.__op = tile(self.__col, self.__row, loc=loc, ip=ip)

        @property
        def op(self) -> TileOp:
            assert self.__op != None
            return self.__op

        @op.setter
        def op(self, op: TileOp):
            assert self.__op == None
            self.__op = op

    def __init__(self, cols: int, rows: int) -> None:
        self.__cols = cols
        self.__rows = rows
        self.__tiles: list[list[MyDevice.__MyDeviceTile]] = []
        for c in range(self.__cols):
            self.__tiles.append([])
            for r in range(self.__rows):
                self.__tiles[c].append(MyDevice.__MyDeviceTile(c, r))

    @property
    def rows(self) -> int:
        return self.__rows

    @property
    def cols(self) -> int:
        return self.__cols

    def resolve_tile(
        self,
        tile: MyTile,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        self.__tiles[tile.col][tile.row].resolve(loc, ip)
        tile.op = self.__tiles[tile.col][tile.row].op


class NPU1Col1(MyDevice):
    def __init__(self) -> None:
        super().__init__(cols=1, rows=4)

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        return AIEDevice.npu1_1col


class NPU1Col4(MyDevice):
    def __init__(self) -> None:
        super().__init__(cols=4, rows=4)

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        return AIEDevice.npu1_1col


class XCVC1902(MyDevice):
    def __init__(self) -> None:
        super().__init__(cols=1, rows=1)  # TODO: this is just dummy for now

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        return AIEDevice.xcvc1902
