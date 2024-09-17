"""
TODOs:
* docs
"""

from abc import ABC, abstractmethod

from .. import ir


class Resolvable(ABC):
    @abstractmethod
    def resolve(
        cls,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None: ...
