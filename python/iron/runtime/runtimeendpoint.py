from __future__ import annotations

from ..dataflow.endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile, PlacementTile


class RuntimeEndpoint(ObjectFifoEndpoint):
    def __init__(self, placement: PlacementTile) -> RuntimeEndpoint:
        self._tile = placement

    @property
    def tile(self) -> PlacementTile:
        return self._tile

    def place(self, tile: Tile) -> None:
        assert not isinstance(
            self._tile, Tile
        ), f"Worker already placed at {self._tile}, cannot place {tile}"
        self._tile = tile

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeEndpoint):
            return NotImplemented
        return self.tile == other.tile

    def __str__(self) -> str:
        return f"RuntimeEndpoint({self.tile})"
