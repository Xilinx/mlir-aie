"""
TODOs:
* docs
* error handling
* tile types"
"""

from ... import ir


class MyTile:
    def __init__(self, col: int, row: int) -> None:
        assert isinstance(col, int)
        assert isinstance(row, int)
        self.col: int = col
        self.row: int = row
        self.op = None
