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

from abc import abstractmethod

from ... import ir
from ...dialects.aie import AIEDevice, tile

from ..resolvable import Resolvable
from .tile import MyTile


class MyDevice(Resolvable):
    class MyDeviceTile(Resolvable):
        def __init__(self, column: int, row: int) -> None:
            assert isinstance(column, int)
            assert isinstance(row, int)
            self.__column: int = column
            self.__row: int = row
            self.__op: int = None

        @property
        def op(self):
            assert self.__op != None
            return self.__op

        def resolve(
            self,
            loc: ir.Location = None,
            ip: ir.InsertionPoint = None,
            context: ir.Context = None,
        ) -> None:
            if self.__op != None:
                return
            self.__op = tile(self.__column, self.__row, loc=loc, ip=ip)

    @property
    @abstractmethod
    def rows(self) -> int: ...

    @property
    @abstractmethod
    def columns(self) -> int: ...

    @abstractmethod
    def resolve_tile(
        self,
        col: int,
        row: int,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None: ...


class NPU1Col1(MyDevice):
    def __init__(self) -> None:
        self.__cols: int = 1
        self.__rows: int = 4
        self.__tiles: list[list[MyDevice.MyDeviceTile]] = []
        for c in range(self.__cols):
            self.__tiles.append([])
            for r in range(self.__rows):
                self.__tiles[c].append(MyDevice.MyDeviceTile(c, r))

    @property
    def rows(self) -> int:
        return self.__rows

    @property
    def columns(self) -> int:
        return self.__cols

    def resolve_tile(
        self,
        tile: MyTile,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        self.__tiles[tile.col][tile.row].resolve(loc, ip, context)
        tile.op = self.__tiles[tile.col][tile.row].op

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        return AIEDevice.npu1_1col
