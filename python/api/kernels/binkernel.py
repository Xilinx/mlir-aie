"""
TODO: 
* docs
* types for inout_types
"""

import numpy as np
from typing import get_origin

from ... import ir  # type: ignore

from ...extras.util import np_dtype_to_mlir_type
from ...extras.dialects.ext.func import FuncOp, call  # type: ignore
from ...dialects.aie import external_func
from .kernel import Kernel


class BinKernel(Kernel):
    def __init__(
        self,
        name: str,
        bin_name: str,
        inout_types: list[type[np.ndarray] | np.dtype] = [],
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
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self.__op == None:
            self.__op = external_func(self.__name, inputs=self.__inout_types)

    def __call__(self, *args, **kwargs):
        assert self.__op, "Need to resolve BinKernel before it can be called"
        call(self.__op, args, **kwargs)
