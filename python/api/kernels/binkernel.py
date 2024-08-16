"""
TODO: 
* docs
* types for inout_types
"""

import numpy as np
from typing import get_origin

from ... import ir

from ...extras.util import np_dtype_to_mlir_type
from ...dialects.func import FuncOp
from ...dialects.aie import external_func, call
from ...extras.util import np_ndarray_type_to_mlir_type
from .kernel import MyKernel


class BinKernel(MyKernel):
    def __init__(
        self,
        name: str,
        bin_name: str,
        inout_types: list[
            np.ndarray[np.generic.dtype, np.generic.shape] | np.dtype
        ] = [],
    ) -> None:
        self.__name = name
        self.__bin_name = bin_name
        self.__inout_types = inout_types
        self.__op: FuncOp | None = None

    @property
    def bin_name(self) -> str:
        return self.__bin_name

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
    ) -> None:
        if self.__op == None:
            resolved_inout_types = []
            for t in self.__inout_types:
                if get_origin(t) == np.ndarray:
                    dtype = np_ndarray_type_to_mlir_type(t)
                else:
                    dtype = np_dtype_to_mlir_type(t)
                resolved_inout_types.append(dtype)
            self.__op = external_func(self.__name, inputs=resolved_inout_types)

    def __call__(self, *args, **kwargs):
        assert self.__op, "Need to resolve BinKernel before it can be called"
        call(self.__name, args, kwargs)
