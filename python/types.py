import ctypes
import warnings
from functools import partial
from typing import Union

import numpy as np
from .ir import (
    Attribute,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E4M3B11FNUZType,
    IndexType,
    IntegerType,
    MemRefType,
    NoneType,
    OpaqueType,
    RankedTensorType,
    Type,
    UnrankedMemRefType,
    UnrankedTensorType,
    VectorType,
    StridedLayoutAttr,
)

from .dialects import transform
from .dialects import pdl

_index = lambda: IndexType.get()
_bool = lambda: IntegerType.get_signless(1)

_i8 = lambda: IntegerType.get_signless(8)
_i16 = lambda: IntegerType.get_signless(16)
_i32 = lambda: IntegerType.get_signless(32)
_i64 = lambda: IntegerType.get_signless(64)

_si8 = lambda: IntegerType.get_signed(8)
_si16 = lambda: IntegerType.get_signed(16)
_si32 = lambda: IntegerType.get_signed(32)
_si64 = lambda: IntegerType.get_signed(64)

_ui8 = lambda: IntegerType.get_unsigned(8)
_ui16 = lambda: IntegerType.get_unsigned(16)
_ui32 = lambda: IntegerType.get_unsigned(32)
_ui64 = lambda: IntegerType.get_unsigned(64)

_f16 = lambda: F16Type.get()
_f32 = lambda: F32Type.get()
_f64 = lambda: F64Type.get()
_bf16 = lambda: BF16Type.get()

_f8e5m2 = lambda: Float8E5M2Type.get()
_f8e4m3 = lambda: Float8E4M3FNType.get()
_f8e4m3b11fnuz = lambda: Float8E4M3B11FNUZType.get()

_cmp16 = lambda: ComplexType.get(_f16())
_cmp32 = lambda: ComplexType.get(_f32())
_cmp64 = lambda: ComplexType.get(_f64())

_none = lambda: NoneType.get()

_pdl_operation = lambda: pdl.OperationType.get()

opaque = lambda dialect_namespace, buffer: OpaqueType.get(dialect_namespace, buffer)


def _transform_any_op():
    return transform.AnyOpType.get()


def _llvm_ptr():
    return Type.parse("!llvm.ptr")


def placeholder_opaque():
    return opaque("scf", "placeholder")


_name_to_type = {
    "index": _index,
    "bool": _bool,
    "i8": _i8,
    "i16": _i16,
    "i32": _i32,
    "i64": _i64,
    "si8": _si8,
    "si16": _si16,
    "si32": _si32,
    "si64": _si64,
    "ui8": _ui8,
    "ui16": _ui16,
    "ui32": _ui32,
    "ui64": _ui64,
    "f16": _f16,
    "f32": _f32,
    "f64": _f64,
    "bf16": _bf16,
    "f8e5m2": _f8e5m2,
    "f8e4m3": _f8e4m3,
    "f8e4m3b11fnuz": _f8e4m3b11fnuz,
    "cmp16": _cmp16,
    "cmp32": _cmp32,
    "cmp64": _cmp64,
    "none": _none,
    "pdl_operation": _pdl_operation,
    "transform_any_op": _transform_any_op,
    "llvm_ptr": _llvm_ptr,
}


def __getattr__(name):
    if name in _name_to_type:
        return _name_to_type[name]()
    # this kicks it to the default module attribute lookup (i.e., functions defined below and such)
    return None


_np_dtype_to_mlir_type_ctor = {
    np.int8: _i8,
    np.int16: _i16,
    np.int32: _i32,
    # windows
    np.intc: _i32,
    np.int64: _i64,
    # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: _index,
    np.uintp: _index,
    np.float16: _f16,
    np.float32: _f32,
    np.float64: _f64,
}

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)


_mlir_type_to_ctype = {
    _bool: ctypes.c_bool,
    _i8: ctypes.c_byte,
    _i64: ctypes.c_int,
    _f32: ctypes.c_float,
    _f64: ctypes.c_double,
}


def mlir_type_to_ctype(mlir_type):
    __mlir_type_to_ctype = {k(): v for k, v in _mlir_type_to_ctype.items()}
    return _mlir_type_to_ctype.get(mlir_type)


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
        return _bool()
    elif isinstance(py_val, int):
        if -(2 ** 31) <= py_val < 2 ** 31:
            return _i32()
        elif 2 ** 31 <= py_val < 2 ** 32:
            return _ui32()
        elif -(2 ** 63) <= py_val < 2 ** 63:
            return _i64()
        elif 2 ** 63 <= py_val < 2 ** 64:
            return _ui64()
        else:
            raise RuntimeError(f"Nonrepresentable integer {py_val}.")
    elif isinstance(py_val, float):
        if (
            abs(py_val) == float("inf")
            or abs(py_val) == 0.0
            or py_val != py_val  # NaN
            or np.finfo(np.float32).min <= abs(py_val) <= np.finfo(np.float32).max
        ):
            return _f32()
        else:
            return _f64()
    elif isinstance(py_val, np.ndarray):
        dtype = np_dtype_to_mlir_type(py_val.dtype.type)
        return RankedTensorType.get(py_val.shape, dtype)
    else:
        raise NotImplementedError(
            f"Unsupported Python value {py_val=} with type {type(py_val)}"
        )


def shaped(*args, element_type: Type = None, type_constructor=None):
    if type_constructor is None:
        raise ValueError("shaped is an abstract base class - cannot be constructed")
    if (element_type is None and args and not isinstance(args[-1], Type)) or (
        args and isinstance(args[-1], Type) and element_type is not None
    ):
        raise ValueError(
            f"either element_type must be provided explicitly XOR last arg to tensor type constructor must be the element type"
        )
    if element_type is not None:
        type = element_type
        sizes = args
    else:
        type = args[-1]
        sizes = args[:-1]
    if sizes:
        return type_constructor(sizes, type)
    else:
        return type_constructor(type)


def vector(*args, element_type: Type = None):
    return shaped(*args, element_type=element_type, type_constructor=VectorType.get)


def tensor(*args, element_type: Type = None):
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped(
            *args, element_type=element_type, type_constructor=UnrankedTensorType.get
        )
    else:
        return shaped(
            *args, element_type=element_type, type_constructor=RankedTensorType.get
        )


def memref(
    *args,
    element_type: Type = None,
    memory_space: int = None,
    layout: tuple[tuple[int, ...], int] = None,
):
    if memory_space is None:
        memory_space = 0
    if layout is not None:
        strides, offset = layout
        layout = StridedLayoutAttr.get(offset, strides)
    memory_space = Attribute.parse(str(memory_space))
    if not len(args) or len(args) == 1 and isinstance(args[-1], Type):
        return shaped(
            *args,
            element_type=element_type,
            type_constructor=partial(UnrankedMemRefType.get, memory_space=memory_space),
        )
    else:
        return shaped(
            *args,
            element_type=element_type,
            type_constructor=partial(
                MemRefType.get, memory_space=memory_space, layout=layout
            ),
        )


def memref_type_to_np_dtype(memref_type):
    _memref_type_to_np_dtype = {
        memref(element_type=_f16()): np.float16,
        memref(element_type=_f32()): np.float32,
        memref(_f64()): np.float64,
        memref(element_type=_bool()): np.bool_,
        memref(_i8()): np.int8,
        memref(_i32()): np.int32,
        memref(_i64()): np.int64,
    }
    return _memref_type_to_np_dtype.get(memref_type)


def transform_op(name):
    return transform.OperationType.get(name)
