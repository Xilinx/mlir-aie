"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import ABC, abstractmethod
from ..phys.tile import MyTile


class MyObjectFifoEndpoint(ABC):
    @abstractmethod
    def get_tile(self) -> MyTile: ...
