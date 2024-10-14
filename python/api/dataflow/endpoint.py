"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import abstractmethod
from ..phys.tile import Tile
from ..resolvable import Resolvable


class ObjectFifoEndpoint(Resolvable):
    @property
    @abstractmethod
    def tile(self) -> Tile | None: ...
