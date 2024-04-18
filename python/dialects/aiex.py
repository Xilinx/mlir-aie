# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
from functools import partial
import itertools
from operator import itemgetter
from typing import Union

import numpy as np

from ._aiex_ops_gen import *
from . import aie
from .aie import (
    DMAChannelDir,
    LockAction,
    Neighbors,
    TileOp,
    find_matching_buffers,
    find_matching_flows,
    find_matching_locks,
    find_neighbors,
)
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..ir import DictAttr, IntegerAttr, UnitAttr

# noinspection PyUnresolvedReferences
from ..extras.dialects.ext import memref


# Comes from _aie
register_dialect(get_dialect_registry())

ipu_sync = partial(ipu_sync, column_num=1, row_num=1)


class NpuDmaMemcpyNd(NpuDmaMemcpyNdOp):
    """Specialize NpuDmaMemcpyNdOp class constructor to take python integers"""

    def __init__(
        self,
        metadata,
        bd_id,
        mem,
        offsets: MixedValues = None,
        sizes: MixedValues = None,
        strides: MixedValues = None,
    ):
        x = 0
        y = 0
        if offsets is None:
            offsets = [0] * 4
        if sizes is None:
            sizes = [0] * 4
        if strides is None:
            strides = [0] * 3
        dynamic_offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(
            offsets
        )
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
        dynamic_strides, _packed_strides, static_strides = _dispatch_mixed_values(
            strides
        )
        super().__init__(
            x,
            y,
            mem,
            dynamic_offsets,
            dynamic_sizes,
            dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            metadata,
            bd_id,
        )


ipu_dma_memcpy_nd = NpuDmaMemcpyNd


_PROLOG = [
    0x00000011,
    0x01000405,
    0x01000100,
    0x0B590100,
    0x000055FF,
    0x00000001,
    0x00000010,
    0x314E5A5F,
    0x635F5F31,
    0x676E696C,
    0x39354E5F,
    0x6E693131,
    0x5F727473,
    0x64726F77,
    0x00004573,
    0x07BD9630,
    0x000055FF,
]

# from runtime_lib/xaiengine/aie-rt/driver/src/global/xaiemlgbl_params.h
# these aren't completely correct - right values but not necessarily the right names?
XAIEMLGBL_NOC_MODULE_DMA_MM2S_0_TASK_QUEUE = 0x0001D214
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE = 0x0001D204
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK = 0x80000000
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK = 0x0000000F
XAIEMLGBL_CORE_MODULE_CORE_CONTROL = 0x00032000

# from dpufw/include/RunInstOpt.h
SHIM_DMA_BD0_BASE_ADDR = 0x1D000
SHIM_BD_OFFSET = 0x20
# from dpufw/include/dpu_func.h
DDR_AIE_ADDR_OFFSET = 0x80000000


def _get_prolog():
    return _PROLOG[:]


# based on https://github.com/Xilinx/mlir-aie/blob/cb232a43383ef3b8efd8b408545c9b74885578ad/lib/Targets/AIETargetNPU.cpp
def _ipu_sync(column, row=0, direction=0, channel=0, column_num=1, row_num=1):
    if isinstance(channel, IntegerAttr):
        channel = int(channel)
    words = [None] * 2
    op_code = 3
    words[0] = (op_code & 0xFF) << 24
    words[0] |= (column & 0xFF) << 16
    words[0] |= (row & 0xFF) << 8
    words[0] |= direction & 0x1

    words[1] = (channel & 0xFF) << 24
    words[1] |= (column_num & 0xFF) << 16
    words[1] |= (row_num & 0xFF) << 8
    assert not any(w is None for w in words)
    return words


def _ipu_write32(column, row, address, value):
    words = [None] * 3
    op_code = 2
    words[0] = (op_code & 0xFF) << 24
    words[0] |= (column & 0xFF) << 16
    words[0] |= (row & 0xFF) << 8
    words[1] = address
    words[2] = value
    assert not any(w is None for w in words)
    return words


def _ipu_shimtile_push_queue(channel_dir, channel_index, column, bd_id, repeats=0):
    if isinstance(channel_index, IntegerAttr):
        channel_index = int(channel_index)
    if channel_dir == DMAChannelDir.MM2S:
        address = XAIEMLGBL_NOC_MODULE_DMA_MM2S_0_TASK_QUEUE
    else:
        address = XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE
    if channel_index == 1:
        address += 0x8
    value = bd_id & XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK
    value |= (repeats & 0xFF) << 16
    if channel_dir == DMAChannelDir.S2MM:
        # issue token
        value |= XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK

    row = 0
    return _ipu_write32(column, row, address, value)


# based on ExecWriteBdExtendShimTileOpt @ dpufw/src/include/RunInstOpt.h:666
def _exec_write_bd_extend_shim_tile_opt(iptr, tensor_addr):
    bd_id = iptr[0] & 0x0000000F
    column = (iptr[0] & 0x00FF0000) >> 16
    buffer_offset = iptr[3]
    # upper 16 bits are for packets...
    tensor_addr += buffer_offset + DDR_AIE_ADDR_OFFSET
    word3 = tensor_addr & 0xFFFFFFFC
    word4 = (iptr[4] & 0xFFFF0000) | (tensor_addr >> 32)

    write_addr = SHIM_DMA_BD0_BASE_ADDR + (bd_id * SHIM_BD_OFFSET)
    row = 0
    words = [
        *_ipu_write32(column, row, write_addr, iptr[2]),
        *_ipu_write32(column, row, write_addr + 4, word3),
        *_ipu_write32(column, row, write_addr + 8, word4),
        *_ipu_write32(column, row, write_addr + 12, iptr[5]),
        *_ipu_write32(column, row, write_addr + 16, iptr[6]),
        *_ipu_write32(column, row, write_addr + 20, iptr[7]),
        *_ipu_write32(column, row, write_addr + 24, iptr[8]),
        *_ipu_write32(column, row, write_addr + 28, iptr[9]),
    ]
    return words


def _update_tensor_addr_shim_tile(column, bd_id, tensor_addr, buffer_offset=0):
    # note upper 16 bits are for packets and thus this clears them
    tensor_addr += buffer_offset + DDR_AIE_ADDR_OFFSET
    word3 = tensor_addr & 0xFFFFFFFC
    word4 = 0xFFFF0000 | (tensor_addr >> 32)

    write_addr = SHIM_DMA_BD0_BASE_ADDR + (bd_id * SHIM_BD_OFFSET)
    row = 0
    words = [
        *_ipu_write32(column, row, write_addr + 4, word3),
        *_ipu_write32(column, row, write_addr + 8, word4),
    ]
    return words


# corresponds to ExecWriteBdExtendShimTileOpt
def _ipu_writebd_shimtile(
    column,
    bd_id,
    buffer_length,
    buffer_offset=0,
    ddr_id=0,
    d2_stride=1,
    d1_size=None,
    d1_stride=1,
    d0_size=None,
    d0_stride=1,
    iteration_size=0,
    iteration_stride=0,
    iteration_current=0,
    lock_acq_enable=0,
    lock_acq_id=0,
    lock_acq_val=0,
    lock_rel_id=0,
    lock_rel_val=0,
    next_bd=0,
    use_next_bd=0,
    data_width=32,
):
    d2_stride -= 1
    d1_stride -= 1
    d0_stride -= 1
    assert d2_stride >= 0 and d1_stride >= 0 and d0_stride >= 0
    # byte offset
    buffer_offset *= data_width // 8
    # None means do not wrap which is 0 on the arch
    if d1_size is None:
        d1_size = 0
    # None means do not wrap which is 0 on the arch
    if d0_size is None:
        d0_size = 0

    column_num = 1
    enable_packet = 0
    out_of_order_id = 0
    packet_id = 0
    packet_type = 0

    words = [None] * 10
    op_code = 6
    words[0] = (op_code & 0xFF) << 24
    words[0] |= (column & 0xFF) << 16
    words[0] |= (column_num & 0xFF) << 8
    words[0] |= (ddr_id & 0xF) << 4
    words[0] |= bd_id & 0xF

    # TODO: Address Incr
    words[1] = 0
    words[2] = buffer_length
    # addr_low in the spec/docs
    words[3] = buffer_offset
    # words[4] = addr_high & 0x0000FFFF

    # En Packet , OoO BD ID , Packet ID , Packet Type
    words[4] = (enable_packet & 0x1) << 30
    words[4] |= (out_of_order_id & 0x3F) << 24
    words[4] |= (packet_id & 0x1F) << 19
    words[4] |= (packet_type & 0x7) << 16

    # TODO: Secure Access
    words[5] = (d0_size & 0x3FF) << 20
    words[5] |= d0_stride & 0xFFFFF

    # burst length;
    words[6] = 0x80000000
    words[6] |= (d1_size & 0x3FF) << 20
    words[6] |= d1_stride & 0xFFFFF

    # TODO: SIMID, AxCache, AXQoS
    words[7] = d2_stride & 0xFFFFF

    words[8] = (iteration_current & 0x3F) << 26
    words[8] |= (iteration_size & 0x3F) << 20
    words[8] |= iteration_stride & 0xFFFFF

    # TODO: TLAST Suppress
    valid_bd = 1
    words[9] = (next_bd & 0xF) << 27
    words[9] |= (use_next_bd & 0x1) << 26
    words[9] |= (valid_bd & 0x1) << 25
    words[9] |= (lock_rel_val & 0xEF) << 18
    words[9] |= (lock_rel_id & 0xF) << 13
    words[9] |= (lock_acq_enable & 0x1) << 12
    words[9] |= (lock_acq_val & 0xEF) << 5
    words[9] |= lock_acq_id & 0xF

    assert not any(w is None for w in words)

    return words


def _ipu_noop():
    words = [None] * 1
    op_code = 0
    words[0] = (op_code & 0xFF) << 24
    return words


def _ipu_core_enable(column, row):
    # note this clears the reset bit
    return _ipu_write32(column, row, XAIEMLGBL_CORE_MODULE_CORE_CONTROL, 1)


class ipu:
    noop = _ipu_noop
    write32 = _ipu_write32
    shimtile_push_queue = _ipu_shimtile_push_queue
    writebd_shimtile = _ipu_writebd_shimtile
    sync = _ipu_sync
    get_prolog = _get_prolog
    enable_cores = _ipu_core_enable
    _exec_write_bd_extend_shim_tile_opt = _exec_write_bd_extend_shim_tile_opt
    _update_tensor_addr_shim_tile = _update_tensor_addr_shim_tile


def process_bd(
    acq_lock,
    buffer,
    rel_lock,
    *,
    acq_action=LockAction.AcquireGreaterEqual,
    rel_action=LockAction.Release,
    acq_val=None,
    rel_val=None,
    offset=None,
    len=None,
    dimensions=None,
):
    aie.use_lock(acq_lock, acq_action, value=acq_val)
    aie.dma_bd(buffer, offset=offset, len=len, dimensions=dimensions)
    aie.use_lock(rel_lock, rel_action, value=rel_val)


def send_bd(channel, *args, **kwargs):
    @aie.dma(DMAChannelDir.MM2S, channel)
    def d():
        process_bd(*args, **kwargs)


def receive_bd(channel, *args, **kwargs):
    @aie.dma(DMAChannelDir.S2MM, channel)
    def d():
        process_bd(*args, **kwargs)


def forward_bd(
    tile,
    buffer,
    s2mm_channel_idx,
    *,
    mm2s_channel_idx=None,
    read_in_lock=None,
    write_out_lock=None,
    repeat_count=None,
):
    if isinstance(s2mm_channel_idx, IntegerAttr):
        s2mm_channel_idx = int(s2mm_channel_idx)
    if isinstance(mm2s_channel_idx, IntegerAttr):
        mm2s_channel_idx = int(mm2s_channel_idx)
    if mm2s_channel_idx is None:
        mm2s_channel_idx = s2mm_channel_idx
    buffer_sym_name = buffer.owner.opview.sym_name
    if buffer_sym_name:
        buffer_sym_name = buffer_sym_name.value
    if read_in_lock is None:
        read_in_lock = aie.lock(
            tile,
            init=1,
            sym_name=(f"{buffer_sym_name}_read_in_lock" if buffer_sym_name else None),
        )
    if write_out_lock is None:
        write_out_lock = aie.lock(
            tile,
            init=0,
            sym_name=(f"{buffer_sym_name}_write_out_lock" if buffer_sym_name else None),
        )

    loop = repeat_count is None

    @aie.dma(DMAChannelDir.S2MM, s2mm_channel_idx, loop=loop, repeat_count=repeat_count)
    def dma_incoming():
        process_bd(read_in_lock, buffer, write_out_lock)

    @aie.dma(DMAChannelDir.MM2S, mm2s_channel_idx, loop=loop, repeat_count=repeat_count)
    def dma_outgoing():
        process_bd(write_out_lock, buffer, read_in_lock)


@contextmanager
def hold_lock(
    acq_lock,
    rel_lock,
    *,
    acq_action=LockAction.AcquireGreaterEqual,
    acq_val=None,
    acq_en=None,
    rel_action=LockAction.Release,
    rel_val=None,
):
    aie.use_lock(acq_lock, acq_action, value=acq_val, acq_en=acq_en)
    try:
        yield
    finally:
        aie.use_lock(rel_lock, rel_action, value=rel_val)


class Channel:
    def __init__(
        self,
        tile,
        buffer=None,
        shape=None,
        dtype=None,
        buffer_name=None,
        initial_value=None,
        producer_lock_id=None,
        producer_lock_init=None,
        producer_lock_sym_name=None,
        producer_lock_annot=None,
        consumer_lock_id=None,
        consumer_lock_init=None,
        consumer_lock_sym_name=None,
        consumer_lock_annot=None,
        loc=None,
        ip=None,
    ):
        if buffer is None:
            assert (
                shape is not None and dtype is not None
            ), f"must provide either existing buffer or buffer shape and dtype"
            buffer = aie.buffer(
                tile,
                shape,
                dtype,
                name=buffer_name,
                initial_value=initial_value,
                loc=loc,
                ip=ip,
            )

        self.buffer = buffer
        if producer_lock_sym_name is None:
            producer_lock_sym_name = (
                f"{buffer.get_name().replace('%', '')}_producer_lock"
            )
        self.producer_lock = aie.lock(
            tile,
            lock_id=producer_lock_id,
            init=producer_lock_init,
            sym_name=producer_lock_sym_name,
            annot=producer_lock_annot,
            loc=loc,
            ip=ip,
        )
        if consumer_lock_sym_name is None:
            consumer_lock_sym_name = (
                f"{buffer.get_name().replace('%', '')}_consumer_lock"
            )
        self.consumer_lock = aie.lock(
            tile,
            lock_id=consumer_lock_id,
            init=consumer_lock_init,
            sym_name=consumer_lock_sym_name,
            annot=consumer_lock_annot,
            loc=loc,
            ip=ip,
        )

    @contextmanager
    def put(self, *, acq_val=None, rel_val=None):
        with hold_lock(
            self.producer_lock, self.consumer_lock, acq_val=acq_val, rel_val=rel_val
        ):
            yield self.buffer

    @contextmanager
    def get(self, *, acq_val=None, rel_val=None):
        with hold_lock(
            self.consumer_lock, self.producer_lock, acq_val=acq_val, rel_val=rel_val
        ):
            yield self.buffer

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.buffer=} {self.producer_lock=} {self.consumer_lock=}>".replace(
            "self.", ""
        )


def _is_nd_list_of_tuples(thing):
    if isinstance(thing, list):
        return _is_nd_list_of_tuples(thing[0])
    if isinstance(thing, tuple):
        return np.dtype(type(thing[0])).char, len(thing)
    return None


# this is dumb but it's specfically to handle the special but also common case of [(K,)]
def _convert_nd_list_of_tuples_to_np(thing):
    def _get_outer_shape(thing):
        if isinstance(thing, list):
            return [len(thing)] + _get_outer_shape(thing[0])
        return []

    def _itemgetter(thing, idx):
        if len(idx) == 1:
            return thing[idx[0]]
        else:
            _itemgetter(thing[0], idx[1:])

    shape = _get_outer_shape(thing)
    res = np.empty(shape, dtype=object)
    for idx, _ in np.ndenumerate(res):
        res[idx] = _itemgetter(thing, idx)

    return res


def _broadcast_args_to(args, dest_shape=None):
    args = list(args)
    for i, arg in enumerate(args):
        maybe_dtype_char_inner_len = _is_nd_list_of_tuples(arg)
        if maybe_dtype_char_inner_len is not None:
            dtype_char, inner_len = maybe_dtype_char_inner_len
            if inner_len == 1:
                arg = _convert_nd_list_of_tuples_to_np(arg)
            else:
                dtype = f"{dtype_char}," * inner_len
                arg = np.array(arg, dtype=dtype).astype(object)
        arg = np.core.shape_base._atleast_nd(arg, len(dest_shape))
        assert (
            np.broadcast_shapes(arg.shape, dest_shape) == dest_shape
        ), f"Only broadcasting from source to dest is supported: {arg=} {arg.shape=} {dest_shape=}"
        args[i] = np.broadcast_to(arg, dest_shape)
    return args


class TileArray:
    def __init__(self, cols=5, rows=6, df=None):
        if df is None:
            if isinstance(cols, int):
                cols = list(range(cols))
            if isinstance(rows, int):
                rows = list(range(rows))
            assert isinstance(cols, (list, tuple)) and isinstance(rows, (list, tuple))
            df = np.array([[aie.tile(c, r) for r in rows] for c in cols])
        self.df = df
        self.channels = np.empty_like(df, dtype=object)

    @property
    def tile(self):
        if isinstance(self.df, TileOp):
            return self.df
        assert len(self.df) == 1, f"convenience accessor only for getting a single tile"
        return self.df[0]

    def flow(self, other, *args, **kwargs):
        return broadcast_flow(self.df, other.df, *args, **kwargs)

    def channel(self, *args, **kwargs):
        args = _broadcast_args_to((self.df,) + args, self.shape)
        kwargs = dict(
            zip(kwargs.keys(), _broadcast_args_to(kwargs.values(), self.shape))
        )
        r = np.vectorize(Channel, otypes=[object])(*args, **kwargs)
        if r.size == 1:
            r = r[0]
        return r

    def __rshift__(self, other):
        return broadcast_flow(self.df, other.df)

    def __lshift__(self, other):
        r = np.frompyfunc(partial(broadcast_flow), 2, 1).outer(other.df, self.df)
        if isinstance(r, np.ndarray):
            r = r.flatten().tolist()
            if len(r) == 1:
                r = r[0]
        return r

    def __getitem__(self, item):
        # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
        # numpy advanced indexing is a little mind-binding:
        # self.df[[[0], [1]], 0].shape == (2, 1)
        # the way it works is _the indices_ are broadcasted too
        # "Advanced indices always are broadcast and iterated as one"

        # canonicalize slices so that below shortcut works correctly
        if isinstance(item, int):
            item = [item]
        item = list(item)
        for j, i in enumerate(item):
            if isinstance(i, slice):
                item[j] = list(range(*i.indices(self.df.shape[j])))
        # take a shortcut and turn something like self.df[[0, 1], WHATEVER] into self.df[[[0], [1]], WHATEVER]
        # i.e. outer dim will match the length of the first idx
        if len(item) == 2:
            if np.asarray(item[0]).ndim == 1:
                item[0] = np.asarray(item[0])[:, np.newaxis]
            if np.asarray(item[1]).ndim == 1:
                item[1] = np.asarray(item[1])[np.newaxis, :]
        item = tuple(item)
        return TileArray(df=self.df[item])

    def __contains__(self, item):
        if isinstance(self.df, np.ndarray):
            return item in self.df
        assert isinstance(self.df, TileOp)
        return item == self.df

    def flows(self, **kwargs):
        return find_matching_flows(self, **kwargs)

    def locks(self, **kwargs):
        return find_matching_locks(self, **kwargs)

    def buffers(self, **kwargs):
        return find_matching_buffers(self, **kwargs)

    def buffer(self, *args, **kwargs):
        args = _broadcast_args_to((self.df,) + args, self.shape)
        kwargs = dict(
            zip(kwargs.keys(), _broadcast_args_to(kwargs.values(), self.shape))
        )
        r = np.vectorize(aie.buffer, otypes=[object])(*args, **kwargs)
        if r.size == 1:
            r = r[0]
        return r

    def lock(self, *args, **kwargs):
        args = _broadcast_args_to((self.df,) + args, self.shape)
        kwargs = dict(
            zip(kwargs.keys(), _broadcast_args_to(kwargs.values(), self.shape))
        )
        r = np.vectorize(aie.lock, otypes=[object])(*args, **kwargs)
        if r.size == 1:
            r = r[0]
        return r

    def __iter__(self):
        for idx, v in np.ndenumerate(self.df):
            if v is not None:
                yield idx, TileArray(df=np.array([v]))

    def neighbors(self, logical=True):
        r = np.vectorize(
            lambda tile: Neighbors(
                **{
                    k: TileArray(df=np.array([v])) if v is not None else None
                    for k, v in find_neighbors(tile, logical=logical).__dict__.items()
                }
            ),
            otypes=[object],
        )(self.df)

        if r.size == 1:
            r = r[0]
        return r

    @property
    def shape(self):
        if isinstance(self.df, TileOp):
            return (1,)
        return self.df.shape

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.df}>"


def broadcast_flow(
    source: Union[np.ndarray, TileOp],
    dest: Union[np.ndarray, TileOp],
    source_bundle=None,
    source_channel=None,
    dest_bundle=None,
    dest_channel=None,
    source_annot=None,
    dest_annot=None,
):
    if isinstance(source, TileOp):
        source = np.asarray([source])
    if isinstance(dest, TileOp):
        dest = np.asarray([dest])
    for chan in [source_channel, dest_channel]:
        assert chan is None or np.all(
            np.array(chan) != None
        ), "can't handle mixed auto channel assignment"

    def _find_next_channel(used_channels):
        max_used_channel = max(used_channels, default=-1)
        for i in range(max_used_channel):
            if i not in used_channels:
                channel = i
                break
        else:
            channel = max_used_channel + 1
        return channel

    if source_channel is None or np.all(np.array(source_channel) == None):
        source_channel = np.empty_like(source, dtype=None)
        for s, indices in zip(*map(list, np.unique(source, return_index=True))):
            matching_flows = find_matching_flows([s], filter_source=True)
            used_channels = set(int(f.source_channel) for f in matching_flows)
            source_channel.flat[indices] = _find_next_channel(used_channels)

    if dest_channel is None or np.all(np.array(dest_channel) == None):
        used_channels = {}
        for d in np.unique(dest):
            matching_flows = find_matching_flows([d], filter_dest=True)
            used_channels[d] = set(int(f.dest_channel) for f in matching_flows)
        dest_channel = np.empty_like(dest, dtype=None)
        for idx, dst in np.ndenumerate(dest):
            dest_channel[idx] = _find_next_channel(used_channels[dst])
            used_channels[dst].add(dest_channel[idx])

    args = _broadcast_args_to(
        [
            source,
            source_bundle,
            source_channel,
            dest,
            dest_bundle,
            dest_channel,
            source_annot,
            dest_annot,
        ],
        dest.shape,
    )
    for i, arg in enumerate(args):
        args[i] = arg.flatten()
    flows = []
    for _grp_name, grp in itertools.groupby(zip(*args), key=itemgetter(0)):
        for g in grp:
            *flow_args, source_annot, dest_annot = g
            flow_ = aie.flow(*flow_args)
            if source_annot is not None:
                flow_.attributes["source_annot"] = DictAttr.get(
                    {source_annot: UnitAttr.get()}
                )
            if dest_annot is not None:
                flow_.attributes["dest_annot"] = DictAttr.get(
                    {dest_annot: UnitAttr.get()}
                )
            flows.append(flow_)
    if len(flows) == 1:
        flows = flows[0]
    return flows
