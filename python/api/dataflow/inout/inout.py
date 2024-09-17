"""
TODO: 
* docs
"""

from abc import abstractmethod
from typing import Callable

from .... import ir
from ....extras.util import np_dtype_to_mlir_type
from ....dialects.aiex import runtime_sequence
from ...tensor import MyTensorType
from ..endpoint import MyObjectFifoEndpoint
from ..objectfifo import ObjectFifoHandle
from ...phys.tile import MyTile


class InOutProgram(MyObjectFifoEndpoint):
    @abstractmethod
    def get_fifos(self) -> list[ObjectFifoHandle]: ...


class MyInOutProgram(InOutProgram):
    def __init__(
        self,
        sequence_fn: Callable[..., None],
        inout_types: list[MyTensorType],
        fifos=list[ObjectFifoHandle],
        coords: tuple[int, int] = (0, 0),  # TODO: how to get default
    ):
        self.sequence_fn = sequence_fn
        self.inout_types = inout_types
        self.tile = MyTile(coords[0], coords[1])
        for f in fifos:
            f.set_endpoint(self)
        self.fifos = fifos

    def get_tile(self) -> MyTile:
        assert self.tile != None
        return self.tile

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self.fifos

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        my_memref_types = [t.memref_type for t in self.inout_types]
        @runtime_sequence(*my_memref_types)
        def sequence(*args, **kwargs):
            self.sequence_fn(*args, *self.fifos, **kwargs)
