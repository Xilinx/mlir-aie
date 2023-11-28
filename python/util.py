from typing import Union, Optional, Sequence

import numpy as np

from .dialects import arith
from .dialects import complex
from .dialects.linalg.opdsl.lang.emitter import (
    _is_floating_point_type,
    _is_complex_type,
)
from .extras import types as T
from .ir import (
    DenseElementsAttr,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    RankedTensorType,
    Type,
    Value,
)


_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,
    # windows
    np.intc: T.i32,
    np.int64: T.i64,
    # is technically wrong i guess but numpy by default casts python scalars to this
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


def constant(
    value: Union[int, float, bool, np.ndarray],
    type: Optional[Type] = None,
    index: Optional[bool] = None,
    *,
    loc: Location = None,
    ip: InsertionPoint = None,
) -> Value:
    """Instantiate arith.constant with value `value`.

    Args:
      value: Python value that determines the value attribute of the
        arith.constant op.
      type: Optional MLIR type that type of the value attribute of the
        arith.constant op; if omitted the type of the value attribute
        will be inferred from the value.
      index: Whether the MLIR type should be an index type; if passed the
        type argument will be ignored.

    Returns:
      ir.OpView instance that corresponds to instantiated arith.constant op.
    """
    if index is not None and index:
        type = IndexType.get()
    if type is None:
        type = infer_mlir_type(value)

    assert type is not None

    if _is_complex_type(type):
        value = complex(value)
        return complex.ConstantOp(
            type,
            list(
                map(
                    lambda x: FloatAttr.get(type.element_type, x),
                    [value.real, value.imag],
                )
            ),
            loc=loc,
            ip=ip,
        )

    if _is_floating_point_type(type) and not isinstance(value, np.ndarray):
        value = float(value)

    if RankedTensorType.isinstance(type) and isinstance(value, (int, float, bool)):
        ranked_tensor_type = RankedTensorType(type)
        value = np.full(
            ranked_tensor_type.shape,
            value,
            dtype=mlir_type_to_np_dtype(ranked_tensor_type.element_type),
        )

    if isinstance(value, np.ndarray):
        value = DenseElementsAttr.get(
            value,
            type=type,
        )

    return arith.ConstantOp(type, value, loc=loc, ip=ip)
