# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
import itertools
from operator import itemgetter

import numpy as np

from ._aiex_ops_gen import *
from ._aiex_ops_gen import (
    npu_write32 as _npu_write32,
    npu_maskwrite32 as _npu_maskwrite32,
    npu_sync as _npu_sync,
    npu_address_patch as _npu_address_patch,
    npu_rtp_write as _npu_rtp_write,
    npu_push_queue as _npu_push_queue,
)
from ._aie_ops_gen import ObjectFifoCreateOp, EndOp, RuntimeSequenceOp
from . import aie
from .aie import (
    DMAChannelDir,
    LockAction,
    Neighbors,
    TileOp,
    bds,
    dma_bd,
    _as_bd_i32,
    _as_bd_i64_dims,
    _as_i32,
)
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..helpers.npdtypes import v8bfp16ebs8, v16bfp16ebs16
from ..ir import (
    DictAttr,
    IntegerAttr,
    UnitAttr,
    Type,
    InsertionPoint,
    Attribute,
    AttrBuilder,
)

# noinspection PyUnresolvedReferences
from ..extras import types as T
from ..extras.dialects import arith
from ..helpers.util import try_convert_np_type_to_mlir_type
from ..helpers.taplib import TensorAccessPattern

# Comes from _aie
register_dialect(get_dialect_registry())


def npu_write32(address, value, buffer=None, column=None, row=None, **kwargs):
    return _npu_write32(
        _as_i32(address),
        _as_i32(value),
        buffer=buffer,
        column=column,
        row=row,
        **kwargs,
    )


def npu_maskwrite32(address, value, mask, buffer=None, column=None, row=None, **kwargs):
    return _npu_maskwrite32(
        _as_i32(address),
        _as_i32(value),
        _as_i32(mask),
        buffer=buffer,
        column=column,
        row=row,
        **kwargs,
    )


def npu_sync(column, row, direction, channel, column_num=1, row_num=1, **kwargs):
    return _npu_sync(
        _as_i32(column),
        _as_i32(row),
        _as_i32(direction),
        _as_i32(channel),
        _as_i32(column_num),
        _as_i32(row_num),
        **kwargs,
    )


def npu_address_patch(addr, arg_idx, arg_plus, **kwargs):
    return _npu_address_patch(addr, _as_i32(arg_plus), arg_idx=arg_idx, **kwargs)


def npu_rtp_write(buffer, index, value, **kwargs):
    return _npu_rtp_write(buffer, index, _as_i32(value), **kwargs)


def npu_push_queue(
    column, row, direction, channel, issue_token, repeat_count, bd_id, **kwargs
):
    return _npu_push_queue(
        column,
        row,
        direction,
        channel,
        issue_token,
        _as_i32(repeat_count),
        _as_i32(bd_id),
        **kwargs,
    )


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
        axcache (optional): The raw 4-bit AxCACHE value for the DMA's AXI-MM transfers. If
            omitted, the target model's default AxCACHE value is used.

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
        axcache: int | None = None,
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
            _as_bd_i64_dims(offsets, "npu_dma_memcpy_nd offsets")
        )
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(
            _as_bd_i64_dims(sizes, "npu_dma_memcpy_nd sizes")
        )
        dynamic_strides, _packed_strides, static_strides = _dispatch_mixed_values(
            _as_bd_i64_dims(strides, "npu_dma_memcpy_nd strides")
        )
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
            axcache=axcache,
            packet=packet,
            offset_parameter=offset_parameter,
        )


npu_dma_memcpy_nd = NpuDmaMemcpyNd


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


def _task_dims(sizes, strides):
    """Normalize a single-BD task's dimensions and derive its repeat count.

    Returns ``(sizes, strides, repeat_count, repeat_count_val)``. Every
    dimension before the last three is an iteration dimension rather than a
    transferred extent: the task runs once per index of them, so its repeat
    count is their product less one, while the transferred length is
    ``prod(sizes[-3:])`` (see ``shim_dma_bd``).

    Fewer than 4 dimensions are left-padded with unit dimensions. Without that,
    a 3-dim ``sizes[0] > 1`` would count both as an access dimension and as the
    repeat count, so the task would re-issue the whole transfer ``sizes[0]``
    times and ``dma_await_task`` would never return.

    A BD holds 4 dimensions. ``aie-decompose-large-dma-bd`` splits the ones past
    that off into further descriptors, which it can only do for constant sizes
    and strides. A constant repeat count folds to the ``repeat_count``
    attribute; a runtime one (4 dimensions at most) flows into the
    ``repeat_count_val`` operand, in i32, the queue field's width.
    """
    repeat_count = 0
    repeat_count_val = None
    if sizes is None:
        return sizes, strides, repeat_count, repeat_count_val
    sizes = list(sizes)
    if strides is not None:
        strides = list(strides)
    while len(sizes) < 4:
        sizes = [1] + sizes
        if strides is not None:
            strides = [0] + strides
    # The BD block lowers only constants, so widen to the i64 operand type here.
    sizes = _as_bd_i64_dims(sizes, "sizes")
    strides = _as_bd_i64_dims(strides, "strides")

    def constant(v):
        return isinstance(v, (int, np.integer))

    outer = sizes[:-3]
    if len(sizes) > 4 and not all(map(constant, sizes + (strides or []))):
        raise ValueError(
            f"a DMA BD with more than 4 dimensions (got {len(sizes)}) needs "
            "constant sizes and strides, which the compiler splits into BDs of 4"
        )
    if all(map(constant, outer)):
        runs = int(np.prod([int(v) for v in outer]))
        if runs > 1:
            repeat_count = runs - 1
    else:
        # sizes may be i64 (DynamicIndexList); narrow before subtracting.
        repeat_count_val = _as_bd_i32(outer[0]) - _as_i32(1)
    return sizes, strides, repeat_count, repeat_count_val


def shim_dma_bd(
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    burst_length: int = 0,
    axcache: int | None = None,
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
        strides = [0] * (len(sizes) - 1) + [1]

    if transfer_len is None:
        transfer_len = np.prod(sizes[-3:])

    dma_bd(
        mem,
        sizes=sizes,
        strides=strides,
        offset=offset,
        transfer_len=transfer_len,
        burst_length=burst_length,
        axcache=axcache,
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
    axcache: int | None = None,
    packet: tuple[int] | None = None,
    offset_parameter: str | None = None,
):
    """_summary_
    Enables data transfers between the AIE Engine array and external memory.
    DMA tasks operations do not require to specify a BD number and are capable of chaining BD operations.

    Args:
        alloc: The alloc argument associates the DMA task with an ObjectFIFO. This argument is called alloc because the shim-side end of a data transfer (specifically a channel on a shim tile) is referenced through a so-called "shim DMA allocation". When an ObjectFIFO is created with a Shim Tile endpoint, an allocation with the same name as the ObjectFIFO is automatically generated.
        mem: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to.
        tap (optional): A TensorAccessPattern is an alternative method of specifying offset/sizes/strides for determining an access pattern over the mem buffer.
        offset (optional): Starting point for the data transfer. Default values is 0. A runtime i64 value is narrowed to the i32 the field takes, as is ``transfer_len``.
        sizes: The extent of data to be transferred across each dimension. The dimensions before the last three are iteration dimensions, one execution of the BD per index; past four in all, which must then be constant, the compiler splits the transfer into several BDs.
        strides (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
        issue_token (optional): If a token is issued, one may call dma_await_task on the returned task. Default is False.
        burst_length (optional): The configuration of the burst length for the DMA task. If 0, defaults to the highest available value.
        axcache (optional): The raw 4-bit AxCACHE value for the DMA's AXI-MM transfers. If
            omitted, the target model's default AxCACHE value is used.
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

    sizes, strides, repeat_count, repeat_count_val = _task_dims(sizes, strides)
    if transfer_len is None and sizes is not None:
        transfer_len = np.prod(sizes[-3:])
    offset, transfer_len = _as_bd_i32(offset), _as_bd_i32(transfer_len)
    task = dma_configure_task_for(
        alloc,
        repeat_count=repeat_count,
        repeat_count_val=repeat_count_val,
        issue_token=issue_token,
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
                axcache=axcache,
                packet=packet,
                offset_parameter=offset_parameter,
            )
            EndOp()
    return task


def tile_dma_single_bd_task(
    tile,
    direction,
    channel,
    buffer,
    offset=None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len=None,
    issue_token: bool = False,
    packet: tuple[int, int] | None = None,
    bd_id: int | None = None,
    acquire: tuple = (),
    release: tuple = (),
):
    """Configure and return a DMA task on a mem tile or core tile channel.

    The non-shim sibling of
    [`shim_dma_single_bd_task`][aiex.shim_dma_single_bd_task]. Where that one
    reaches its shim channel through an objectFIFO's shim DMA allocation and
    moves a host buffer, this one names a tile and channel directly and moves a
    buffer that lives on that tile -- only a shim BD can address DDR.

    ``sizes``/``strides``/``offset``/``transfer_len`` entries may be runtime
    SSA values, which is what lets a mem tile descriptor be rebuilt per
    dispatch. The dynamic encoder cannot infer a length from the buffer's
    shape, so when any of them is runtime an omitted ``transfer_len`` defaults
    to the product of the last three ``sizes``, which must then be given.

    ``acquire``/``release`` take ``(lock, action, value)`` tuples emitted
    around the BD, for handing the buffer to or from a compute tile.

    Args:
        tile: the tile whose DMA channel this task runs on.
        direction: ``DMAChannelDir.S2MM`` or ``DMAChannelDir.MM2S``.
        channel: hardware channel index. On a mem tile this also decides which
            half of the BD pool ``bd_id`` may come from -- an even channel
            reaches only the low half, an odd channel only the high half. A
            string names an ``aie.route_endpoint`` on ``tile`` instead, whose
            channel the compiler assigns.
        buffer: an ``aie.buffer`` on ``tile``.
        sizes, strides: the access pattern, outermost dimension first. Give
            both or neither: there is no default stride to pair with a size.
        issue_token: issue a completion token, so ``dma_await_task`` may wait
            on the returned task.
        bd_id: pin the buffer descriptor id instead of letting
            ``aie-assign-runtime-sequence-bd-ids`` choose one.
    """
    if (sizes is None) != (strides is None):
        raise ValueError(
            "tile_dma_single_bd_task needs sizes and strides together, got "
            f"sizes={sizes} and strides={strides}"
        )
    sizes, strides, repeat_count, repeat_count_val = _task_dims(sizes, strides)
    if (
        transfer_len is None
        and sizes is not None
        and not all(
            isinstance(v, (int, np.integer)) for v in [*sizes, *strides, offset or 0]
        )
    ):
        transfer_len = _as_bd_i32(np.prod(sizes[-3:]))
    task_kwargs = dict(
        repeat_count=repeat_count,
        repeat_count_val=repeat_count_val,
        issue_token=issue_token,
    )
    if isinstance(channel, str):
        task = dma_configure_task_for(channel, **task_kwargs)
    else:
        task = dma_configure_task(tile, direction, channel, **task_kwargs)
    bd_kwargs = {}
    if bd_id is not None:
        bd_kwargs["bd_id"] = bd_id
    if packet is not None:
        bd_kwargs["packet"] = packet
    with bds(task) as bd:
        with bd[0]:
            if acquire:
                aie.use_lock(acquire[0], acquire[1], value=acquire[2])
            dma_bd(
                buffer,
                sizes=sizes,
                strides=strides,
                offset=offset if offset is not None else 0,
                transfer_len=transfer_len,
                **bd_kwargs,
            )
            if release:
                aie.use_lock(release[0], release[1], value=release[2])
            EndOp()
    return task


_orig_dma_await_task = dma_await_task


def dma_await_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_await_task must receive at least one DMAConfigureTaskForOp to wait for"
        )
    for dma_task in args:
        _orig_dma_await_task(task=dma_task)


_orig_dma_free_task = dma_free_task


def dma_free_task(*args: DMAConfigureTaskForOp):
    if len(args) == 0:
        raise ValueError(
            "dma_free_task must receive at least one DMAConfigureTaskForOp to free"
        )
    for dma_task in args:
        _orig_dma_free_task(dma_task)


_orig_dma_start_task = dma_start_task


def dma_start_task(
    *args: DMAConfigureTaskForOp,
    repeat_count: int | None = None,
    no_token: bool = False,
):
    """Push each task onto its channel's queue.

    ``repeat_count`` replaces the task's own count for these starts only, and
    ``no_token`` withholds the completion token the task would otherwise issue.
    A count beyond what one queue push carries is issued as several pushes by
    ``aie-assign-runtime-sequence-bd-ids``.
    """
    if len(args) == 0:
        raise ValueError(
            "dma_start_task must receive at least one DMAConfigureTaskForOp to start"
        )
    for dma_task in args:
        _orig_dma_start_task(
            dma_task, repeat_count=repeat_count, no_token=no_token or None
        )


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

    For example:

    ```python
    val = aiex.read_scratchpad_parameter("foo", T.bf16())
    ```
    """
    return _orig_read_scratchpad_parameter(result_type, name)
