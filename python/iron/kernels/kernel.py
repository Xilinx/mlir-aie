"""
TODO: 
* docs
"""

from abc import abstractmethod
from ..resolvable import Resolvable


class Kernel(Resolvable):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass
