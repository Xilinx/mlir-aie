# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
from functools import partial
import itertools
from operator import itemgetter

import numpy as np

from ._aiex_ops_gen import *
from ._aie_ops_gen import ObjectFifoCreateOp, dma_bd, EndOp
from . import aie
from .aie import (
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
    IntegerAttr,
    UnitAttr,
    Type,
    InsertionPoint,
    Attribute,
    AttrBuilder,
)

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


def shim_dma_bd(
    mem,
    tap: TensorAccessPattern | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    burst_length: int = 0,
    packet: tuple[int] | None = None,
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

    if transfer_len is None:
        transfer_len = np.prod(sizes[-3:])

    dimensions = list(zip(sizes, strides))
    dma_bd(
        mem,
        offset=offset,
        len=transfer_len,
        dimensions=dimensions,
        burst_length=burst_length,
        packet=packet,
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
    if sizes and sizes[0] > 1:
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
