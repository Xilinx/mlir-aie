"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import abstractmethod
from ..phys.tile import Tile, PlacementTile


class ObjectFifoEndpoint:
    @property
    @abstractmethod
    def tile(self) -> PlacementTile: ...

    @abstractmethod
    def place(self, tile: Tile) -> None: ...
