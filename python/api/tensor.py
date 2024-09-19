from functools import cached_property
import numpy as np
import typing

from ..extras.util import np_dtype_to_mlir_type
from ..dialects.memref import MemRefType


class MyTensorType:
    def __init__(self, np_ndarray_type: np.ndarray[np.generic.dtype, np.generic.shape]):
        args = typing.get_args(np_ndarray_type)
        self.__dtype: np.generic.dtype = args[0]
        self.__shape: np.generic.shape = args[1]
        self.__my_numpy_type = np_ndarray_type

    @classmethod
    def get_memref_type(
        cls, np_ndarray_type: np.ndarray[np.generic.dtype, np.generic.shape]
    ) -> MemRefType:
        args = typing.get_args(np_ndarray_type)
        dtype: np.generic.dtype = args[0]
        shape: np.generic.shape = args[1]
        return MemRefType.get(shape=shape, element_type=np_dtype_to_mlir_type(dtype))

    @cached_property
    def memref_type(self) -> MemRefType:
        return MemRefType.get(
            shape=self.__shape, element_type=np_dtype_to_mlir_type(self.__dtype)
        )

    @property
    def shape(self) -> np.generic.shape:
        return self.__shape

    @property
    def dtype(self) -> np.generic.dtype:
        return self.__dtype

    def __eq__(self, other):
        # TODO: may want to be equal to numpy datatypes as well??
        if other:
            return self.__my_numpy_type == other.__my_numpy_type
        return False
