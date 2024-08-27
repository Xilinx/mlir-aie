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
from ...dialects.aie import AIEDevice

from ..resolvable import Resolvable
from .tile import MyTile


class MyDevice(Resolvable):
    @property
    @abstractmethod
    def rows(self) -> int: ...

    @property
    @abstractmethod
    def columns(self) -> int: ...

    @abstractmethod
    def get_tile(self, col: int, row: int) -> MyTile: ...


class NPU1Col1(MyDevice):
    def __init__(self) -> None:
        self.__cols: int = 1
        self.__rows: int = 4
        self.__tiles: list[list[MyTile]] = []
        for c in range(self.__cols):
            self.__tiles.append([])
            for r in range(self.__rows):
                self.__tiles[c].append(MyTile(c, r))

    @property
    def rows(self) -> int:
        return self.__rows

    @property
    def columns(self) -> int:
        return self.__cols

    def get_tile(self, col: int, row: int) -> MyTile:
        return self.__tiles[col][row]

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        return AIEDevice.npu1_1col
