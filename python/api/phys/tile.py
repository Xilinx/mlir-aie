"""
TODOs:
* docs
* error handling
* tile types"
"""

from ... import ir
from ...dialects.aie import tile

from ..resolvable import Resolvable


class MyTile(Resolvable):
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
            pass
        self.__op = tile(self.__column, self.__row, loc=loc, ip=ip)
