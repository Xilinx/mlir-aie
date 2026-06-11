# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
from functools import partial
import itertools
from operator import itemgetter

import numpy as np

from ._aiex_ops_gen import *
from ._aie_ops_gen import ObjectFifoCreateOp, EndOp, RuntimeSequenceOp
from . import aie
from .aie import (
    dma_bd,
    DMAChannelDir,
    LockAction,
    Neighbors,
    TileOp,
    bds,
)
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..helpers.util import v8bfp16ebs8, v16bfp16ebs16
from ..ir import (
    DictAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    UnitAttr,
    Type,
    Value,
    InsertionPoint,
    Attribute,
    AttrBuilder,
)
from . import arith as _arith

# noinspection PyUnresolvedReferences
from ..extras import types as T
from ..helpers.util import try_convert_np_type_to_mlir_type
from ..helpers.taplib import TensorAccessPattern

# Comes from _aie
register_dialect(get_dialect_registry())

npu_sync = partial(npu_sync, column_num=1, row_num=1)


def dma_wait(*args: ObjectFifoCreateOp | str):
    if len(args) == 0:
        raise ValueError(
            "dma_wait must receive at least one dma_meta information to wait for"
        )
    for dma_meta in args:
        str_name = dma_meta
        if isinstance(dma_meta, ObjectFifoCreateOp):
            str_name = dma_meta.sym_name.value
        npu_dma_wait(str_name)


def _cast_to_i64(v):
    """Coerce an SSA Value to i64 for NPU DMA descriptor operands.

    The npu.dma_memcpy_nd op currently declares offsets/sizes/strides as
    Variadic<I64>. Front-end arithmetic, however, may naturally produce
    `index` (from scf.for induction vars), `i32` (from runtime_sequence
    args), or any other signless-integer width. This helper inserts the
    canonical conversion at the call site so callers can pass in whatever
    SSA value falls out of their computation without manual casts.

    TODO(future PR): once AIEX.td tightens the operand type to
    Variadic<I32> (matching the NPU descriptor register width), switch
    this to cast to i32 instead. AIEDmaToNpu's getAsValue already does
    width coercion in either direction, so the lowering does not care
    which width the IR carries.
    """
    if not isinstance(v, Value):
        return v
    i64 = IntegerType.get_signless(64)
    vt = v.type
    if vt == i64:
        return v
    if isinstance(vt, IndexType):
        return _arith.index_cast(i64, v)
    if isinstance(vt, IntegerType):
        if vt.width > 64:
            return _arith.trunci(i64, v)
        return _arith.extui(i64, v)
    raise TypeError(
        f"npu_dma_memcpy_nd offsets/sizes/strides must be index or signless "
        f"integer, got {vt}"
    )


class NpuDmaMemcpyNd(NpuDmaMemcpyNdOp):
    """
    Enables data transfers between the AIE Engine array and external memory.

    Args:
        metadata: This is a reference to the object FIFO or the string name of an object FIFO that records a Shim Tile and one of its DMA channels allocated for the host-side memory transfer. In order to associate the memcpy operation with an object FIFO, this metadata string needs to match the object FIFO name string.
        bd_id: Identifier integer for the particular Buffer Descriptor control registers used for this memcpy. A buffer descriptor contains all information needed for a DMA transfer described in the parameters below.
        mem: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to.
        tap (optional): A TensorAccessPattern is an alternative method of specifying offset/sizes/strides for determining an access pattern over the mem buffer.
        offsets (optional): Start points for data transfer in each dimension. There is a maximum of four offset dimensions.
        sizes: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
        strides (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
        burst_length (optional): The configuration of the burst length for the DMA task. If 0, defaults to the highest available value.

    Note:
        Contiguous row-major access patterns are automatically folded to canonical linear form
        by the compiler's canonicalization pass. For example, a 2D image access
        ``sizes=[1, 1, height, width], strides=[0, 0, width, 1]`` is equivalent to
        ``sizes=[1, 1, 1, height*width], strides=[0, 0, 0, 1]`` and will be canonicalized
        to the latter. This means the natural multidimensional form can always be used
        without concern for the hardware d0 dimension size limit.

    Example:

        npu_dma_memcpy_nd(of_in, 0, input_buffer, sizes=[1, 1, 1, 30])

        The example above describes a linear transfer of 30 data elements, or 120 Bytes, from the input_buffer in host memory into an object FIFO with matching
        metadata labeled "of_in".
        The size dimensions are expressed right to left where the right is dimension 0 and the left dimension 3. Higher dimensions not used should be set to 1.
    """

    def __init__(
        self,
        metadata: str | ObjectFifoCreateOp,
        bd_id,
        mem,
        tap: TensorAccessPattern | None = None,
        offsets: MixedValues | None = None,
        sizes: MixedValues | None = None,
        strides: MixedValues | None = None,
        issue_token: bool | None = None,
        burst_length: int = 0,
        packet: tuple[int] | None = None,
        offset_parameter: str | None = None,
    ):
        if tap and not (offsets is None and sizes is None and strides is None):
            raise ValueError(
                "NpuDmaMemcpyNd can take either a TileAccessPattern OR (sizes and/or strides and/or offsets), but not both."
            )
        if tap:
            sizes = tap.sizes.copy()
            strides = tap.strides.copy()
            # For some reason, the type checking of offsets does not mesh well with offset being a property
            # so here we make sure it is evaluated and properly is seen as an integer.
            offsets = [0] * 3 + [int(tap.offset)]
        else:
            if offsets is None:
                offsets = [0] * 4
            if sizes is None:
                sizes = [0] * 4
            if strides is None:
                strides = [0] * 3 + [1]
        dynamic_offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(
            offsets
        )
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
        dynamic_strides, _packed_strides, static_strides = _dispatch_mixed_values(
            strides
        )
        # The op operands $offsets/$sizes/$strides are Variadic<I64>.
        # Whatever the user's SSA
        # arithmetic produced (index from scf.for, i64 from a wider compute,
        # iN from a custom path), normalise it to i32 here so callers do not
        # have to insert arith.index_cast / trunci / extui themselves.
        dynamic_offsets = [_cast_to_i64(v) for v in dynamic_offsets]
        dynamic_sizes = [_cast_to_i64(v) for v in dynamic_sizes]
        dynamic_strides = [_cast_to_i64(v) for v in dynamic_strides]
        if isinstance(metadata, ObjectFifoCreateOp):
            metadata = metadata.sym_name.value
        super().__init__(
            mem,
            dynamic_offsets,
            dynamic_sizes,
            dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            metadata,
            bd_id,
            issue_token=issue_token,
            burst_length=burst_length,
            packet=packet,
            offset_parameter=offset_parameter,
        )


npu_dma_memcpy_nd = NpuDmaMemcpyNd


# Dynamic convenience wrappers
# These create unified ops with SSA value operands for runtime-parameterized
# sequences. The static attributes serve as placeholders (0) and the dynamic
# SSA values override them at runtime.


def npu_write32_dynamic(dyn_address, dyn_value, *, buffer=None, column=None, row=None):
    """write32 with SSA value operands for runtime-parameterized sequences."""
    return NpuWrite32Op(
        address=0,
        value=0,
        buffer=buffer,
        column=column,
        row=row,
        dyn_address=dyn_address,
        dyn_value=dyn_value,
    )


def npu_maskwrite32_dynamic(
    dyn_address, dyn_value, dyn_mask, *, buffer=None, column=None, row=None
):
    """maskwrite32 with SSA value operands for runtime-parameterized sequences."""
    return NpuMaskWrite32Op(
        address=0,
        value=0,
        mask=0,
        buffer=buffer,
        column=column,
        row=row,
        dyn_address=dyn_address,
        dyn_value=dyn_value,
        dyn_mask=dyn_mask,
    )


def npu_sync_dynamic(
    dyn_column, dyn_row, dyn_direction, dyn_channel, dyn_column_num, dyn_row_num
):
    """sync with SSA value operands for runtime-parameterized sequences."""
    return NpuSyncOp(
        column=0,
        row=0,
        direction=0,
        channel=0,
        column_num=0,
        row_num=0,
        dyn_column=dyn_column,
        dyn_row=dyn_row,
        dyn_direction=dyn_direction,
        dyn_channel=dyn_channel,
        dyn_column_num=dyn_column_num,
        dyn_row_num=dyn_row_num,
    )


# Delegate to the canonical implementation in aie.py to avoid duplication.
from .aie import npu_write_rtp

npu_rtp_write = npu_write_rtp


# Runtime sequence


def runtime_sequence(*inputs: Type, sym_name=None, context=None):
    def decorator(f):
        name = sym_name if sym_name else f.__name__
        seq_op = RuntimeSequenceOp(sym_name=name)
        my_inputs = []
        for input in inputs:
            my_inputs.append(try_convert_np_type_to_mlir_type(input))
        entry_block = seq_op.body.blocks.append(*my_inputs)
        args = entry_block.arguments
        with InsertionPoint(entry_block):
            f(*args)

    return decorator


# Wrap auto-generated dma_bd to support SSA Value operands.
#
# The TableGen op exposes (in addition to the static offset/len/dimensions
# attributes) optional SSA i64 operands `dyn_offset`, `dyn_len`, `dyn_sizes`,
# `dyn_strides`. Whenever a caller passes an SSA `Value` for offset, len, or
# any element of sizes/strides, we route it to the corresponding dynamic
# operand and clear the conflicting static attribute. The static fast path is
# preserved bit-identically for the all-static case so existing callers
# (object FIFO lowering, static dma_task chains) emit the same IR they always
# have.
#
# All-or-nothing dim semantics: as soon as any element of sizes/strides is an
# SSA Value, every element of both lists is materialised as an SSA i64 (static
# entries become `arith.constant`); the static `dimensions` attribute is then
# omitted. This matches the verifier rule that `dimensions` and
# `dyn_sizes`/`dyn_strides` are mutually exclusive.
_orig_dma_bd = dma_bd
_py_len = len  # capture builtin so the `len=` kwarg below does not shadow it


def dma_bd(
    buffer,
    *,
    offset=0,
    len=None,
    sizes=None,
    strides=None,
    dimensions=None,
    **kwargs,
):
    i64 = IntegerType.get_signless(64)

    # offset routing
    if isinstance(offset, Value):
        dyn_offset = _cast_to_i64(offset)
        static_offset = 0
    else:
        dyn_offset = None
        static_offset = offset

    # len routing
    if isinstance(len, Value):
        dyn_len = _cast_to_i64(len)
        static_len = None
    else:
        dyn_len = None
        static_len = len

    # Collect candidate (size, stride) pairs from either the explicit
    # sizes/strides kwargs or a list of (size, stride) tuples in `dimensions`.
    pairs = None
    if sizes is not None:
        if strides is None:
            raise ValueError("dma_bd: when sizes is given, strides is required")
        if _py_len(sizes) != _py_len(strides):
            raise ValueError("dma_bd: sizes and strides must be the same length")
        pairs = list(zip(sizes, strides))
    elif dimensions is not None:
        # `dimensions` may already be a BDDimLayoutArrayAttr (no SSA Values);
        # in that case nothing to coerce.
        if isinstance(dimensions, Attribute):
            return _orig_dma_bd(
                buffer,
                [],
                [],
                offset=static_offset,
                len=static_len,
                dimensions=dimensions,
                dyn_offset=dyn_offset,
                dyn_len=dyn_len,
                **kwargs,
            )
        pairs = [tuple(d) for d in dimensions]

    if not pairs:
        return _orig_dma_bd(
            buffer,
            [],
            [],
            offset=static_offset,
            len=static_len,
            dimensions=None,
            dyn_offset=dyn_offset,
            dyn_len=dyn_len,
            **kwargs,
        )

    any_dyn_dim = any(isinstance(s, Value) or isinstance(t, Value) for s, t in pairs)
    if not any_dyn_dim:
        return _orig_dma_bd(
            buffer,
            [],
            [],
            offset=static_offset,
            len=static_len,
            dimensions=pairs,
            dyn_offset=dyn_offset,
            dyn_len=dyn_len,
            **kwargs,
        )

    # Static dim values must be materialised as SSA i32 to feed the
    # dyn_sizes/dyn_strides operands. The lowering pass rejects ops other
    # than aie.dma_bd / aie.use_lock / aie.next_bd / aie.end inside the BD
    # block, so we hoist the arith.constants out to live just before the
    # enclosing dma_configure_task in the runtime_sequence body. SSA Values
    # supplied by the caller are assumed to dominate the BD block already.
    cur_block = InsertionPoint.current.block
    bd_parent = cur_block.owner  # dma_configure_task / shim_dma_*_task op
    hoist_ip = InsertionPoint(bd_parent)

    def _coerce(v):
        if isinstance(v, Value):
            return _cast_to_i64(v)
        with hoist_ip:
            return _arith.constant(i64, int(v))

    dyn_sizes = [_coerce(s) for s, _ in pairs]
    dyn_strides = [_coerce(t) for _, t in pairs]
    return _orig_dma_bd(
        buffer,
        dyn_sizes,
        dyn_strides,
        offset=static_offset,
        len=static_len,
        dimensions=None,
        dyn_offset=dyn_offset,
        dyn_len=dyn_len,
        **kwargs,
    )


_orig_dma_configure_task = dma_configure_task


def dma_configure_task(*args, **kwargs):
    return DMAConfigureTaskOp(T.index(), *args, **kwargs)


_orig_dma_configure_task_for = dma_configure_task_for


def dma_configure_task_for(alloc, *args, **kwargs):
    alloc_sym = alloc if isinstance(alloc, str) else alloc.sym_name.value
    return DMAConfigureTaskForOp(T.index(), alloc_sym, *args, **kwargs)


_orig_dma_start_bd_chain = dma_start_bd_chain


def dma_start_bd_chain(symbol, args, tile, direction, channel, *pyargs, **kwargs):
    chain_sym = symbol if isinstance(symbol, str) else symbol.sym_name.value
    return DMAStartBdChainOp(
        T.index(), chain_sym, args, tile, direction, channel, *pyargs, **kwargs
    )


_orig_dma_start_bd_chain_for = dma_start_bd_chain_for


def dma_start_bd_chain_for(symbol, args, alloc, *pyargs, **kwargs):
    chain_sym = symbol if isinstance(symbol, str) else symbol.sym_name.value
    alloc_sym = alloc if isinstance(alloc, str) else alloc.sym_name.value
    return DMAStartBdChainForOp(
        T.index(), chain_sym, args, alloc_sym, *pyargs, **kwargs
    )


def shim_dma_bd(
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    burst_length: int = 0,
    packet: tuple[int] | None = None,
    offset_parameter: str | None = None,
):
    if tap and not (offset is None and sizes is None and strides is None):
        raise ValueError(
            "shim_dma_bd can take either a TensorAccessPattern OR (sizes and/or strides and/or offsets), but not both."
        )

    if tap:
        sizes = tap.sizes.copy()
        strides = tap.strides.copy()
        # For some reason, the type checking of offsets does not mesh well with offset being a property
        # so here we make sure it is evaluated and properly is seen as an integer.
        offset = int(tap.offset)

    if offset is None:
        offset = 0
    if sizes is None:
        sizes = [0] * 4
    if strides is None:
        strides = [0] * 3 + [1]

    # If any of sizes contains SSA Values, we cannot statically compute the
    # transfer length via np.prod; the caller must supply transfer_len in
    # that case (it may itself be an SSA Value).
    if transfer_len is None:
        if any(isinstance(s, Value) for s in sizes):
            raise ValueError(
                "shim_dma_bd: transfer_len must be supplied when sizes contains SSA Values"
            )
        transfer_len = np.prod(sizes[-3:])

    # Pass sizes/strides through as parallel lists so the dma_bd wrapper can
    # detect SSA Values and route them to dyn_sizes/dyn_strides as needed.
    dma_bd(
        mem,
        offset=offset,
        len=transfer_len,
        sizes=list(sizes),
        strides=list(strides),
        burst_length=burst_length,
        packet=packet,
        offset_parameter=offset_parameter,
    )


def shim_dma_single_bd_task(
    alloc,
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    issue_token: bool = False,
    burst_length: int = 0,
    packet: tuple[int] | None = None,
    offset_parameter: str | None = None,
):
    """_summary_
    Enables data transfers between the AIE Engine array and external memory.
    DMA tasks operations do not require to specify a BD number and are capable of chaining BD operations.

    Args:
        alloc: The alloc argument associates the DMA task with an ObjectFIFO. This argument is called alloc becuase the shim-side end of a data transfer (specifically a channel on a shim tile) is referenced through a so-called "shim DMA allocation". When an ObjectFIFO is created with a Shim Tile endpoint, an allocation with the same name as the ObjectFIFO is automatically generated.
        mem: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to.
        tap (optional): A TensorAccessPattern is an alternative method of specifying offset/sizes/strides for determining an access pattern over the mem buffer.
        offset (optional): Starting point for the data transfer. Default values is 0.
        sizes: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
        strides (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
        issue_token (optional): If a token is issued, one may call dma_await_task on the returned task. Default is False.
        burst_length (optional): The configuration of the burst length for the DMA task. If 0, defaults to the highest available value.
        packet (optional): The packet header information represented as a (packet_type, packet_id) tuple.

    Example:
        out_task = shim_dma_single_bd_task(of_out, C, sizes=[1, 1, 1, N], issue_token=True)

        The example above describes a linear transfer of N data elements from the C buffer in host memory into an object FIFO with matching metadata labeled "of_out".
        The sizes dimensions are expressed right to left where the right is dimension 0 and the left dimension 3.
        Higher dimensions not used should be set to 1.
    """
    if tap and not (offset is None and sizes is None and strides is None):
        raise ValueError(
            "shim_dma_single_bd_task can take either a TensorAccessPattern OR (sizes and/or strides and/or offsets), but not both."
        )

    if tap:
        sizes = tap.sizes.copy()
        strides = tap.strides.copy()
        # For some reason, the type checking of offsets does not mesh well with offset being a property
        # so here we make sure it is evaluated and properly is seen as an integer.
        offset = int(tap.offset)

    repeat_count = 0
    if sizes:
        # When sizes[0] is a static int we can derive a repeat_count to fold
        # the outer dimension into BD repetition. With an SSA Value we leave
        # repeat_count = 0 and let the BD's dyn_sizes encode the iteration.
        if not isinstance(sizes[0], Value) and sizes[0] > 1:
            repeat_count = sizes[0] - 1
    task = dma_configure_task_for(
        alloc, repeat_count=repeat_count, issue_token=issue_token
    )
    with bds(task) as bd:
        with bd[0]:
            shim_dma_bd(
                mem,
                offset=offset,
                sizes=sizes,
                strides=strides,
                transfer_len=transfer_len,
                burst_length=burst_length,
                packet=packet,
                offset_parameter=offset_parameter,
            )
            EndOp()
    return task


_orig_dma_await_task = dma_await_task


def dma_await_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_await_task must receive at least one DMAConfigureTaskForOp to wait for"
        )
    for dma_task in args:
        _orig_dma_await_task(dma_task)


_orig_dma_free_task = dma_free_task


def dma_free_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_free_task must receive at least one DMAConfigureTaskForOp to free"
        )
    for dma_task in args:
        _orig_dma_free_task(dma_task)


_orig_dma_start_task = dma_start_task


def dma_start_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_start_task must receive at least one DMAConfigureTaskForOp to free"
        )
    for dma_task in args:
        _orig_dma_start_task(dma_task)


def set_lock_value(lock: aie.LockOp, value: int):
    return set_lock(lock, value)


# Parameter ops

_orig_read_scratchpad_parameter = read_scratchpad_parameter


def read_scratchpad_parameter(
    name: str, result_type: Type
) -> _orig_read_scratchpad_parameter:
    """Read a scratchpad runtime parameter inside an `aie.core` body.

    Args:
        name: The `@sym_name` of the `aiex.scratchpad_parameter` declaration.
        result_type: The MLIR scalar type of the result (e.g. `T.bf16()`, `T.i32()`).

    Returns:
        An SSA value of the given type.

    Example::

        val = aiex.read_scratchpad_parameter("foo", T.bf16())
    """
    return _orig_read_scratchpad_parameter(result_type, name)
