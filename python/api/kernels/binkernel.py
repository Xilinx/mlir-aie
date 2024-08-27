"""
TODO: 
* docs
* types for inout_types
"""

from ... import ir
from ...extras.util import np_dtype_to_mlir_type, get_arg_types
from ...dialects.memref import MemRefType
from ...dialects.aie import external_func, call
from .kernel import MyKernel


class BinKernel(MyKernel):
    def __init__(self, name: str, bin_name: str, inout_types: list) -> None:
        assert isinstance(name, str)
        assert len(name) > 0
        assert isinstance(bin_name, str)
        assert len(bin_name) > 0
        assert isinstance(inout_types, list)
        self.__name = name
        self.__bin_name = bin_name
        self.__inout_types = inout_types
        self.__op = None

    @property
    def bin_name(self) -> str:
        return self.__bin_name

    def resolve(
        self,
        loc: ir.Location = None,
        ip: ir.InsertionPoint = None,
        context: ir.Context = None,
    ) -> None:
        assert self.__op == None
        resolved_inout_types = []
        for t in self.__inout_types:
            try:
                dtype = np_dtype_to_mlir_type(t)
            except Exception:
                dtype = get_arg_types(t)
                if dtype is None:
                    # Interpret as a dummy memref
                    dtype = MemRefType.get(
                        shape=t[0], element_type=np_dtype_to_mlir_type(t[1])
                    )
            resolved_inout_types.append(dtype)
        self.__op = external_func(self.__name, inputs=resolved_inout_types)

    def __call__(self, *args, **kwargs):
        assert self.__op
        call(self.__name, args, kwargs)
