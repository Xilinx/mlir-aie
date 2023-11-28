from typing import Union

import numpy as np

from .extras import types as T
from .ir import (
    F32Type,
    F64Type,
    IntegerType,
    RankedTensorType,
)

_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,  # windows
    np.intc: T.i32,
    np.int64: T.i64,  # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: T.index,
    np.uintp: T.index,
    np.float16: T.f16,
    np.float32: T.f32,
    np.float64: T.f64,
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F32Type, F64Type, RankedTensorType]:
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
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)
