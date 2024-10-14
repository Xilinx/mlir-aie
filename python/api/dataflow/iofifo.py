"""
TODO: 
* docs
* types
* join/distribute
"""

import numpy as np

from .objectfifo import ObjectFifo
from ..io.iocoordinator import IOCoordinator


class IOObjectFifo(ObjectFifo):
    def __init__(
        self,
        depth: int,
        obj_type: type[np.ndarray],
        io_coord: IOCoordinator,
        name: str | None = None,
    ):
        super().__init__(self, depth=depth, obj_type=obj_type, name=name)
        self.io_coord = io_coord

    def fill(self, data_tile, source) -> None:
        self.io_coord.fill(self, data_tile, source)

    def drain(self, data_tile, dest) -> None:
        self.io_coord.drain(self, data_tile, dest)
