"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import abstractmethod
from ..phys.tile import MyTile
from ..resolvable import Resolvable


class MyObjectFifoEndpoint(Resolvable):
    @property
    @abstractmethod
    def tile(self) -> MyTile: ...
