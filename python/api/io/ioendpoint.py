from __future__ import annotations

from ..dataflow.endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile


class IOEndpoint(ObjectFifoEndpoint):
    def __init__(self, column: int, row: int) -> IOEndpoint:
        self.__tile = Tile(column, row)

    @property
    def tile(self) -> Tile | None:
        return self.__tile

    def place(self, tile: Tile) -> None:
        assert not isinstance(
            self.__tile, Tile
        ), f"Worker already placed at {self.tile}, cannot place {tile}"
        self.__tile == tile

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IOEndpoint):
            return NotImplemented
        return self.tile == other.tile

    def __str__(self) -> str:
        return f"IOEndpoint({self.tile})"
