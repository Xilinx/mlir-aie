from functools import cached_property
import numpy as np
from typing import Literal

from ..extras.util import np_dtype_to_mlir_type
from ..dialects.memref import MemRefType


class MyTensorType:
    def __init__(self, dtype: np.generic, shape: np.generic.shape):
        self.__dtype = dtype
        self.__shape = shape
        self.__my_numpy_type = np.ndarray[dtype, Literal[tuple(shape)]]

    @cached_property
    def memref_type(self) -> MemRefType:
        return MemRefType.get(
            shape=self.__shape, element_type=np_dtype_to_mlir_type(self.__dtype)
        )

    @property
    def shape(self) -> np.generic.shape:
        return self.__shape

    @property
    def dtype(self) -> np.generic:
        return self.__dtype

    def __eq__(self, other):
        # TODO: may want to be equal to numpy datatypes as well??
        if other:
            return self.__my_numpy_type == other.__my_numpy_type
        return False
