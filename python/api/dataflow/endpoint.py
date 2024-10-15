"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import abstractmethod
from ..phys.tile import Tile


class ObjectFifoEndpoint:
    @property
    @abstractmethod
    def tile(self) -> Tile | None: ...
