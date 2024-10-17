from collections import defaultdict
import numpy as np
import sys
from typing import Sequence, get_args, get_origin

from ..extras import types as T
from ..ir import (
    F32Type,
    F64Type,
    IntegerType,
    MemRefType,
    OpView,
    RankedTensorType,
    Value,
    VectorType,
)
from ml_dtypes import bfloat16


_np_dtype_to_mlir_type_ctor = defaultdict(
    lambda: None,
    {
        # Signed integer types
        np.int8: T.i8,
        np.int16: T.i16,
        np.int32: T.i32,
        np.intc: T.i32,  # windows
        np.int64: T.i64,
        # Unsigned integer types
        np.uint8: T.ui8,
        np.uint16: T.ui16,
        np.uint32: T.ui32,
        np.uint64: T.ui64,
        # Floating point types
        np.float16: T.f16,
        np.float32: T.f32,
        np.float64: T.f64,
        # Block floating point types
        bfloat16: T.bf16,
        # Index Types
        # this is technically wrong i guess but numpy by default casts python scalars to this
        # so to support passing lists of ints we map to index type
        np.longlong: T.index,
        np.uintp: T.index,
    },
)

NpuDType = (
    np.int8
    | np.int16
    | np.int32
    | np.intc
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.longlong
    | np.uintp
    | bfloat16
)

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    mlir_type = _np_dtype_to_mlir_type_ctor[np_dtype]
    if mlir_type:
        return mlir_type()
    else:
        raise AttributeError(
            "Failed to map np dtype to mlir python type: " + str(np_dtype)
        )


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)


def infer_mlir_type(
    py_val: int | float | bool | np.ndarray, memref=False, vector=False
) -> IntegerType | F32Type | F64Type | RankedTensorType:
    """Infer MLIR type (`ir.Type`) from supported python values.

    Note ints and floats are mapped to 64-bit types.

    Args:
      py_val: Python value that's either a numerical value or numpy array.

    Returns:
      MLIR type corresponding to py_val.
    """
    if isinstance(py_val, bool):
        return T.bool()
    elif isinstance(py_val, int):
        if -(2**31) <= py_val < 2**31:
            return T.i32()
        elif 2**31 <= py_val < 2**32:
            return T.ui32()
        elif -(2**63) <= py_val < 2**63:
            return T.i64()
        elif 2**63 <= py_val < 2**64:
            return T.ui64()
        else:
            raise RuntimeError(f"Nonrepresentable integer {py_val}.")
    elif isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return T.f32()
        else:
            return T.f64()
    elif isinstance(py_val, np.ndarray):
        dtype = np_dtype_to_mlir_type(py_val.dtype.type)
        if memref:
            return MemRefType.get(py_val.shape, dtype)
        elif vector:
            return VectorType.get(py_val.shape, dtype)
        else:
            return RankedTensorType.get(py_val.shape, dtype)
    elif isinstance(py_val, NpuDType):
        return np_dtype_to_mlir_type(type(py_val))
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def memref_type_to_np_dtype(memref_type):
    _memref_type_to_np_dtype = {
        T.memref(element_type=T.f16()): np.float16,
        T.memref(element_type=T.f32()): np.float32,
        T.memref(T.f64()): np.float64,
        T.memref(element_type=T.bool()): np.bool_,
        T.memref(T.i8()): np.int8,
        T.memref(T.i32()): np.int32,
        T.memref(T.i64()): np.int64,
        T.memref(T.bf16()): bfloat16,
    }
    return _memref_type_to_np_dtype.get(memref_type)


def np_ndarray_type_get_shape(ndarray_type: type[np.ndarray]) -> tuple[int, ...]:
    shape = get_args(ndarray_type)[0]
    assert isinstance(shape, tuple), "np.ndarray shape must be a tuple of integers"
    for elem in shape:
        assert isinstance(elem, int), "np.ndarray shape must be a tuple of integers"
    return shape


def np_ndarray_type_get_dtype(ndarray_type: type[np.ndarray]) -> NpuDType:
    return get_args(get_args(ndarray_type)[1])[0]


def np_ndarray_type_to_memref_type(ndarray_type: type[np.ndarray]):
    shape = np_ndarray_type_get_shape(ndarray_type)
    dtype = np_ndarray_type_get_dtype(ndarray_type)
    return T.memref(*shape, element_type=np_dtype_to_mlir_type(dtype))


def try_convert_np_type_to_mlir_type(input_type):
    if get_origin(input_type) == np.ndarray:
        output_type = np_ndarray_type_to_memref_type(input_type)
    elif input_type in get_args(NpuDType):
        output_type = np_dtype_to_mlir_type(input_type)
    else:
        output_type = input_type
    return output_type


def get_arg_types(objs: Sequence[int | float | Value | OpView]):
    my_types = []
    for o in objs:
        if isinstance(o, Value):
            my_types.append(o.type)
        elif isinstance(o, OpView):
            if len(o.results.types) != 1:
                raise AttributeError(
                    f"Operation given to a region op as a parameter ({o}) has more "
                    "than one return type ({o.results.types}), which would lead to a mismatch "
                    "between number of operands and number of operand types"
                )
            my_types += o.results.types
        elif isinstance(o, (int, float)):
            my_types.append(type(o))
        else:
            return None
    return my_types


class DataTileSpec:
    def __init__(self, offset, dimensions, transfer_len):
        self.offset = offset
        self.dimensions = dimensions
        self.sizes = [size for (size, _stride) in dimensions] if dimensions else None
        self.strides = (
            [stride for (_size, stride) in dimensions] if dimensions else None
        )
        self.transfer_len = transfer_len


class DataTiler:
    def __init__(
        self,
        total_data: int,
        initial_offset: int = 0,
        sizes=None,
        strides=None,
        dimensions=None,
    ):
        self.__offset = initial_offset
        self.__dimensions = dimensions
        if dimensions:
            if sizes or strides:
                raise ValueError(
                    f"My only supply either dimensions ({dimensions}) OR sizes ({sizes}) and strides ({strides})"
                )
            self.__dimensions = dimensions
        else:
            if strides and not sizes:
                sizes = [1, 1, 1, total_data]
            elif sizes and not strides:
                strides = [0, 0, 0, 1]

            if not sizes and not sizes:
                self.__len = total_data
                self.__transfer_len = total_data
            else:
                self.__dimensions = list(zip(sizes, strides))

        if self.__dimensions:
            self.__len = np.prod([size for (size, _stride) in self.__dimensions])
            self.__transfer_len = self.__len // self.__dimensions[0][0]
        self.__num_tiles = total_data // self.__len
        self.__tile_num = 0

    def __iter__(self):
        return self

    def __next__(self) -> DataTileSpec:
        if self.__num_tiles > self.__tile_num:
            data_tile_spec = DataTileSpec(
                self.__offset, self.__dimensions, self.__transfer_len
            )
            self.__tile_num += 1
            self.__offset += self.__len
            return data_tile_spec
        else:
            raise StopIteration
