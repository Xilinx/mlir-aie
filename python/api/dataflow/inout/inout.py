"""
TODO: 
* docs
"""

from abc import abstractmethod
import numpy as np
from typing import Callable

from .... import ir
from ....dialects.aiex import runtime_sequence
from ....extras.util import np_ndarray_type_to_mlir_type

from ..objectfifo import ObjectFifoHandle
from ...resolvable import Resolvable


class InOutSequence(Resolvable):
    @abstractmethod
    def get_fifos(self) -> list[ObjectFifoHandle]: ...


class MyInOutSequence(InOutSequence):
    def __init__(
        self,
        sequence_fn: Callable[..., None],
        inout_types: list[np.ndarray[np.generic.dtype, np.generic.shape]],
        fifos=list[ObjectFifoHandle],
    ):
        self.sequence_fn = sequence_fn
        self.inout_types = inout_types
        self.fifos = fifos.copy()

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self.fifos.copy()

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        my_memref_types = [np_ndarray_type_to_mlir_type(t) for t in self.inout_types]

        @runtime_sequence(*my_memref_types)
        def sequence(*args, **kwargs):
            self.sequence_fn(*args, *self.fifos, **kwargs)
