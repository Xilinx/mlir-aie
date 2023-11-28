import re
from typing import Union, Optional, Sequence

from .arith import constant
from .. import memref
from .._ods_common import get_op_result_or_op_results
from ...extras import types as T
from ...ir import MLIRError, ShapedType, Type, Value

S = ShapedType.get_dynamic_size()


def _alloc(
    op_ctor,
    sizes: Sequence[Union[int]],
    element_type: Type,
    *,
    loc=None,
    ip=None,
):
    dynamic_sizes = []
    result_type = T.memref(*sizes, element_type)
    return get_op_result_or_op_results(
        op_ctor(result_type, dynamic_sizes, [], loc=loc, ip=ip)
    )


def alloc(sizes: Sequence[Union[int, Value]], element_type: Type, *, loc=None, ip=None):
    return _alloc(memref.AllocOp, sizes, element_type, loc=loc, ip=ip)


def alloca(
    sizes: Sequence[Union[int, Value]], element_type: Type, *, loc=None, ip=None
):
    return _alloc(memref.AllocaOp, sizes, element_type, loc=loc, ip=ip)


def load(mem: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None):
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(memref.LoadOp(mem, indices, loc=loc, ip=ip))


def store(
    value: Value, mem: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None
):
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(
        memref.StoreOp(value, mem, indices, loc=loc, ip=ip)
    )


def subview(
    source: "MemRef",
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if static_offsets is None:
        static_offsets = []
    if strides is None:
        strides = []
    if static_strides is None:
        static_strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    sizes = []
    wrong_type = T.memref(*static_sizes, source.dtype)
    if offsets and static_offsets:
        assert all(s == S for s in static_offsets)
    if strides and static_strides:
        assert all(s == S for s in static_strides)
    val = memref.subview(
        wrong_type,
        source,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )
    # dumbest hack ever - the default builder doesn't connect to inferReturnTypes
    # but the diag message does
    try:
        val.owner.verify()
        return val
    except MLIRError as e:
        diag = str(e.error_diagnostics[0])
        correct_type = re.findall(r"'memref<(.*)>'", diag)
        assert len(correct_type) == 1
        correct_type = Type.parse(f"memref<{correct_type[0]}>")
        val.owner.erase()
        return memref.subview(
            correct_type,
            source,
            offsets,
            sizes,
            strides,
            static_offsets,
            static_sizes,
            static_strides,
            loc=loc,
            ip=ip,
        )
