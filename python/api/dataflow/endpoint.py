"""
TODO: 
* docs
* types
* producer/consumer
* join
"""

from abc import ABC, abstractmethod
from ..phys.tile import MyTile
from ..resolvable import Resolvable


class MyObjectFifoEndpoint(Resolvable):
    @abstractmethod
    def get_tile(self) -> MyTile: ...
