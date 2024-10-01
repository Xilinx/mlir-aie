from collections import defaultdict
import contextlib
import ctypes
from dataclasses import dataclass
from functools import wraps
import inspect
import numpy as np
import platform
from pathlib import Path
import re
import sys
from typing import Callable, List, Sequence, Tuple, get_args, get_origin

from .meta import op_region_builder
from ..extras import types as T
from ..ir import (
    Block,
    Context,
    F32Type,
    F64Type,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    OpView,
    Operation,
    RankedTensorType,
    SymbolTable,
    Type,
    Value,
    VectorType,
    _GlobalDebug,
)


def is_relative_to(self, other):
    return other == self or other in self.parents


def get_user_code_loc(user_base: Path | None = None):
    from .. import extras

    if Context.current is None:
        return

    mlir_extras_root_path = Path(extras.__path__[0])

    prev_frame = inspect.currentframe().f_back
    if user_base is None:
        user_base = Path(prev_frame.f_code.co_filename)

    while prev_frame.f_back and (
        is_relative_to(Path(prev_frame.f_code.co_filename), mlir_extras_root_path)
        or is_relative_to(Path(prev_frame.f_code.co_filename), sys.prefix)
        or is_relative_to(Path(prev_frame.f_code.co_filename), user_base)
    ):
        prev_frame = prev_frame.f_back
    frame_info = inspect.getframeinfo(prev_frame)
    if sys.version_info.minor >= 11:
        return Location.file(
            frame_info.filename, frame_info.lineno, frame_info.positions.col_offset
        )
    else:
        return Location.file(frame_info.filename, frame_info.lineno, col=0)


@contextlib.contextmanager
def enable_debug():
    _GlobalDebug.flag = True
    yield
    _GlobalDebug.flag = False


def shlib_ext():
    if platform.system() == "Darwin":
        shlib_ext = "dylib"
    elif platform.system() == "Linux":
        shlib_ext = "so"
    elif platform.system() == "Windows":
        shlib_ext = "lib"
    else:
        raise NotImplementedError(f"unknown platform {platform.system()}")

    return shlib_ext


def shlib_prefix():
    if platform.system() in {"Darwin", "Linux"}:
        shlib_pref = "lib"
    elif platform.system() == "Windows":
        shlib_pref = ""
    else:
        raise NotImplementedError(f"unknown platform {platform.system()}")

    return shlib_pref


def find_ops(op, pred: Callable[[OpView, Operation, Module], bool], single=False):
    if isinstance(op, (OpView, Module)):
        op = op.operation

    matching = []

    def find(op):
        if single and len(matching):
            return
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if pred(o):
                        matching.append(o)
                    find(o)

    find(op)
    if single and matching:
        matching = matching[0]
    return matching


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
        # bfloat16: T.bf16,
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
    # | bfloat16
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


_mlir_type_to_ctype = {
    T.bool: ctypes.c_bool,
    T.i8: ctypes.c_byte,
    T.i64: ctypes.c_int,
    T.f32: ctypes.c_float,
    T.f64: ctypes.c_double,
}


def mlir_type_to_ctype(mlir_type):
    __mlir_type_to_ctype = {k(): v for k, v in _mlir_type_to_ctype.items()}
    return _mlir_type_to_ctype.get(mlir_type)


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
    }
    return _memref_type_to_np_dtype.get(memref_type)


def np_ndarray_type_to_memref_type(ndarray_type):
    type_args = get_args(ndarray_type)
    dtype = np_dtype_to_mlir_type(type_args[0])
    shape = type_args[1]
    return T.memref(*shape, element_type=dtype)


def try_convert_np_type_to_mlir_type(input_type):
    if get_origin(input_type) == np.ndarray:
        output_type = np_ndarray_type_to_memref_type(input_type)
    elif input_type in get_args(NpuDType):
        output_type = np_dtype_to_mlir_type(input_type)
    else:
        output_type = input_type
    return output_type


def _get_previous_frame_idents(val, previous_frame):
    return [
        var_name
        for var_name, var_val in previous_frame.f_locals.items()
        if var_val is val
    ]


def _update_caller_vars(previous_frame, args: Sequence, replacements: Sequence):
    """Update caller vars passed as args.

    This function uses CPython API  to update the values
    of the caller's args (not the caller of this function but the caller of caller of this function).
    It does this by searching for a match in the caller's f_locals based on identity (A is A) and then
    updating all corresponding values in the f_locals dict. Finally, it uses PyFrame_LocalsToFast to signal
    to the CPython runtime that an update has been made to f_locals.

    Args:
      previous_frame: The frame in which vars will be updated.
      args: The args to the callee.
      replacements: The values that should replace the values of the vars in the caller.
    """

    if len(args) != len(replacements):
        raise ValueError(f"updates must be 1-1: {args=} {replacements=}")
    # find the name of the iter args in the previous frame
    var_names = [_get_previous_frame_idents(arg, previous_frame) for arg in args]
    for i, var_names in enumerate(var_names):
        for var_name in var_names:
            previous_frame.f_locals[var_name] = replacements[i]
            # signal to update
            # for some reason you can only update one at a time?
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(previous_frame), ctypes.c_int(1)
            )


def make_maybe_no_args_decorator(decorator):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(decorator)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_dec


@dataclass
class Successor:
    op: OpView | Operation
    operands: List[Value]
    block: Block
    pos: int

    def __enter__(self):
        self.bb_ctx_manager = bb(self)
        return self.bb_ctx_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bb_ctx_manager.__exit__(exc_type, exc_val, exc_tb)


@contextlib.contextmanager
def bb(*preds: Tuple[Successor | OpView]):
    current_ip = InsertionPoint.current
    op = current_ip.block.owner
    op_region = op.regions[0]
    args = []
    if len(preds):
        if isinstance(preds[0], (OpView, Successor)):
            args = preds[0].operands
        else:
            raise NotImplementedError(f"{preds[0]=} not supported.")
    arg_locs = list(filter(None, [get_user_code_loc()] * len(args)))
    if len(arg_locs) == 0:
        arg_locs = None
    block = op_region.blocks.append(*[a.type for a in args], arg_locs=arg_locs)
    for p in preds:
        if isinstance(p, OpView):
            p.operation.successors[0] = block
        elif isinstance(p, Successor):
            for i, b in enumerate(p.block.owner.successors):
                if i == p.pos:
                    p.op.successors[i] = block
                    p.block = block
                    break
    with InsertionPoint(block):
        yield block, list(block.arguments)


def region_adder(terminator=None):
    def wrapper(op_region_adder):
        def region_adder_decorator(op, *args, **kwargs):
            if isinstance(op, Value):
                op = op.owner.opview
            region = op_region_adder(op, *args, **kwargs)

            return op_region_builder(op, region, terminator)

        return region_adder_decorator

    return wrapper


class ModuleMeta(type):
    def __new__(cls, name, bases, classdict, **kwargs):
        ip = classdict.pop("ip")
        loc = classdict.pop("loc")
        module_terminator = classdict.pop("module_terminator", None)
        new = super().__new__(cls, name, bases, classdict)
        if module_terminator is not None:
            module_terminator(loc=loc, ip=ip)
        for k, v in classdict.items():
            if callable(v):
                v.qualname = name
        ip.__exit__(None, None, None)
        return new


def find_parent_of_type(test_cb, operation=None):
    if isinstance(operation, OpView):
        operation = operation.operation
    if operation is None:
        parent = InsertionPoint.current.block.owner
    else:
        parent = operation.parent
    for _ in range(10):
        if test_cb(parent):
            return parent
        else:
            parent = parent.parent

    raise RuntimeError("Couldn't matching parent of type")


def is_symbol_table(operation):
    try:
        SymbolTable(operation)
        return True
    except RuntimeError:
        return False


def _get_sym_name(previous_frame, check_func_call=None):
    try:
        with open(inspect.getfile(previous_frame)) as src_file:
            src_lines = src_file.readlines()
            src_line = src_lines[previous_frame.f_lineno - 1].strip()
            ident, func_call = map(lambda x: x.strip(), src_line.split("=", maxsplit=1))
            if check_func_call is not None:
                assert re.match(check_func_call, func_call)
        maybe_unique_sym_name = ident
        symbol_table = SymbolTable(find_parent_of_type(is_symbol_table).operation)
        while maybe_unique_sym_name in symbol_table:
            if re.match(r".*_(\d+)$", maybe_unique_sym_name):
                maybe_unique_sym_name = re.sub(
                    r"(\d+)$", lambda m: str(int(m.group(0)) + 1), maybe_unique_sym_name
                )
            else:
                maybe_unique_sym_name = f"{maybe_unique_sym_name}_0"
        return maybe_unique_sym_name
    except:
        return None


def _unpack_sizes_element_type(sizes_element_type):
    if sizes_element_type[-1] is None:
        sizes_element_type = sizes_element_type[:-1]
    assert isinstance(sizes_element_type[-1], Type)
    element_type = sizes_element_type[-1]
    sizes = sizes_element_type[:-1]
    return sizes, element_type


class getitemproperty:
    def __init__(self, f):
        self.f = f
        self.instance = None

    def __get__(self, instance, _class):
        self.instance = instance
        return self

    def __getitem__(self, item):
        kwargs = {}
        if len(item) > 2:
            # not sure how but you don't need two backs here
            previous_frame = inspect.currentframe().f_back
            for kwarg in item[2:]:
                k = _get_previous_frame_idents(kwarg, previous_frame)
                assert len(k) == 1, f"{len(k)=}"
                kwargs[k[0]] = kwarg
            item = item[:2]

        # f is not a bound method since it was decorated...
        return self.f(self.instance, item, **kwargs)


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
