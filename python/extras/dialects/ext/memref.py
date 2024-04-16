import inspect
from typing import Sequence, Union

import numpy as np

from ._shaped_value import ShapedValue
from .arith import Scalar, constant
from .tensor import _indices_to_indexer, compute_result_shape_reassoc_list
from ... import types as T
from ...meta import region_op
from ...util import (
    _get_sym_name,
    _unpack_sizes_element_type,
    get_user_code_loc,
    infer_mlir_type,
)
from ...._mlir_libs._mlir import register_value_caster
from ....dialects import arith, memref
from ....dialects._ods_common import get_op_result_or_op_results
from ....dialects.memref import *
from ....ir import (
    DenseElementsAttr,
    MemRefType,
    ShapedType,
    Type,
    Value,
    InsertionPoint,
)

S = ShapedType.get_dynamic_size()


def _alloc(
    op_ctor,
    *sizes_element_type: Sequence[Union[int, Value]],
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    sizes, element_type = _unpack_sizes_element_type(sizes_element_type)
    dynamic_sizes = []
    memref_shape = []
    for s in sizes:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            dynamic_sizes.append(s)
    result_type = T.memref(*memref_shape, element_type)

    symbol_operands = []
    return get_op_result_or_op_results(
        op_ctor(result_type, dynamic_sizes, symbol_operands, loc=loc, ip=ip)
    )


def alloc(*sizes: Union[int, Value], element_type: Type = None):
    loc = get_user_code_loc()
    return _alloc(AllocOp, *sizes, element_type, loc=loc, ip=None)


def alloca(*sizes: Union[int, Value], element_type: Type = None):
    loc = get_user_code_loc()
    return _alloc(AllocaOp, *sizes, element_type, loc=loc, ip=None)


def load(mem: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(LoadOp(mem, indices, loc=loc, ip=ip))


def store(
    value: Value, mem: Value, indices: Sequence[Union[Value, int]], *, loc=None, ip=None
):
    if loc is None:
        loc = get_user_code_loc()
    indices = list(indices)
    for idx, i in enumerate(indices):
        if isinstance(i, int):
            indices[idx] = constant(i, index=True)
    return get_op_result_or_op_results(StoreOp(value, mem, indices, loc=loc, ip=ip))


@register_value_caster(MemRefType.static_typeid)
class MemRef(Value, ShapedValue):
    def __str__(self):
        return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx: tuple) -> "MemRef":
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if idx is None:
            return expand_shape(self, (0,), loc=loc)

        idx = list((idx,) if isinstance(idx, (int, slice)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            return load(self, idx, loc=loc)
        else:
            return _subview(self, tuple(idx), loc=loc)

    def __setitem__(self, idx, source):
        loc = get_user_code_loc()

        if not self.has_rank():
            raise ValueError("only ranked memref slicing/indexing supported")

        idx = list((idx,) if isinstance(idx, (Scalar, int, Value)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, Scalar) for d in idx) and len(idx) == len(self.shape):
            assert isinstance(
                source, Scalar
            ), "coordinate insert requires scalar element"
            store(source, self, idx, loc=loc)
        else:
            _copy_to_subview(self, source, tuple(idx), loc=loc)


def expand_shape(
    inp,
    newaxis_dims,
    *,
    loc=None,
    ip=None,
) -> MemRef:
    """Expand the shape of a memref.

    Insert a new axis that will appear at the `axis` position in the expanded
    memref shape.

    Args:
      inp: Input memref-like.
      axis: Position in the expanded axes where the new axis (or axes) is placed.

    Returns:
       View of `a` with the number of dimensions increased.

    """
    if loc is None:
        loc = get_user_code_loc()

    if len(newaxis_dims) == 0:
        return inp

    result_shape, reassoc_list = compute_result_shape_reassoc_list(
        inp.shape, newaxis_dims
    )

    return MemRef(
        memref.expand_shape(
            T.memref(*result_shape, inp.dtype), inp, reassoc_list, loc=loc, ip=ip
        )
    )


def _subview(
    mem: MemRef,
    idx,
    *,
    loc=None,
    ip=None,
) -> MemRef:
    if loc is None:
        loc = get_user_code_loc()

    indexer = _indices_to_indexer(idx, mem.shape)
    out = mem

    if indexer.is_constant():
        out = subview(
            out,
            offsets=indexer.static_offsets(),
            sizes=indexer.static_sizes(),
            strides=indexer.static_strides(),
            loc=loc,
            ip=ip,
        )
    else:
        # special tile case
        offsets = [None] * len(indexer.in_shape)
        static_sizes = [None] * len(indexer.in_shape)
        static_strides = [None] * len(indexer.in_shape)
        for i, ind in enumerate(indexer.indices):
            maybe_size = ind.stop.owner.operands[1]
            if (
                isinstance(ind.start.owner.opview, arith.MulIOp)
                and isinstance(ind.stop.owner.opview, arith.MulIOp)
                and isinstance(ind.stop.owner.operands[0].owner.opview, arith.AddIOp)
                and ind.start.owner.operands[0]
                == ind.stop.owner.operands[0].owner.operands[0]
                and maybe_size.is_constant()
                and isinstance(ind.step, int)
                or isinstance(ind.step, Scalar)
                and ind.step.is_constant()
            ):
                offsets[i] = ind.start
                static_sizes[i] = maybe_size.literal_value
                static_strides[i] = (
                    ind.step.literal_value if isinstance(ind.step, Scalar) else ind.step
                )
            else:
                raise RuntimeError(f"indexing not supported {indexer.indices}")
        offsets = list(filter(None, offsets))
        static_sizes = list(filter(None, static_sizes))
        static_strides = list(filter(None, static_strides))
        assert (
            len(offsets)
            == len(static_sizes)
            == len(static_strides)
            == len(indexer.in_shape)
        ), f"not each slice is statically known: {indexer.indices}"
        out = subview(
            out,
            offsets=offsets,
            sizes=static_sizes,
            strides=static_strides,
            loc=loc,
            ip=ip,
        )

    # This adds newaxis/None dimensions.
    return expand_shape(out, indexer.newaxis_dims, loc=loc, ip=ip)


def _copy_to_subview(
    dest: MemRef,
    source: MemRef,
    idx,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(source, Scalar):
        source = expand_shape(source, (0,), loc=loc, ip=ip)

    dest_subview = _subview(dest, idx, loc=loc, ip=ip)
    assert (
        dest_subview.shape == source.shape
    ), f"Expected matching shape for dest subview {dest_subview.shape} and source {source.shape=}"

    return memref.copy(source, dest_subview, loc=loc, ip=ip)


alloca_scope = region_op(AllocaScopeOp)

_dim = dim


def dim(source, index, *, loc=None, ip=None):
    if isinstance(index, int):
        index = constant(index, index=True)
    return _dim(source=source, index=index, loc=loc, ip=ip)


def global_(
    initial_value=None,
    sym_name=None,
    type_=None,
    sym_visibility="private",
    constant=None,
    alignment=None,
    loc=None,
    ip=None,
):
    if sym_name is None:
        previous_frame = inspect.currentframe().f_back
        sym_name = _get_sym_name(
            previous_frame, check_func_call="memref\\.global_|global_"
        )
        assert (
            sym_name is not None
        ), "couldn't automatically find sym_name in previous frame"
    if loc is None:
        loc = get_user_code_loc()
    if initial_value is None:
        assert type_ is not None
    else:
        assert isinstance(initial_value, np.ndarray)
        type_ = infer_mlir_type(initial_value, memref=True)
        initial_value = DenseElementsAttr.get(
            initial_value,
            type=type_.element_type,
            context=None,
        )
        constant = True

    return memref.global_(
        sym_name,
        type_,
        sym_visibility=sym_visibility,
        initial_value=initial_value,
        constant=constant,
        alignment=alignment,
        loc=loc,
        ip=ip,
    ).opview


def view(source, shape, dtype=None, shift=0):
    if dtype is None:
        dtype = source.type.element_type
    byte_width_dtype = dtype.width // 8
    byte_shift = shift * byte_width_dtype
    byte_shift = constant(byte_shift, index=True)
    return memref.view(T.memref(*shape, dtype), source, byte_shift, [])
