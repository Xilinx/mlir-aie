"""
TODO: 
* docs
"""

from abc import abstractmethod
from ..endpoint import MyObjectFifoEndpoint
from ..objectfifo import ObjectFifoHandle


class InOutProgram(MyObjectFifoEndpoint):

    @abstractmethod
    def get_fifos(self) -> list[ObjectFifoHandle]: ...
