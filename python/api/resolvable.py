"""
TODOs:
* docs
"""

from abc import ABC, abstractmethod

from .. import ir  # type: ignore


class Resolvable(ABC):
    @abstractmethod
    def resolve(
        cls,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...
