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
        self.column: int = column
        self.row: int = row
        self.op: int = None

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        assert self.op == None
        self.op = tile(self.column, self.row, loc=loc, ip=ip)
