# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
from functools import partial

from ._aiex_ops_gen import *
from .aie import DMAChannelDir, LockAction, dma, dma_bd, lock, use_lock
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..ir import IntegerAttr

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Comes from _aie
register_dialect(get_dialect_registry())

ipu_sync = partial(ipu_sync, column_num=1, row_num=1)


class IpuDmaMemcpyNd(IpuDmaMemcpyNdOp):
    """Specialize IpuDmaMemcpyNdOp class constructor to take python integers"""

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


ipu_dma_memcpy_nd = IpuDmaMemcpyNd


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

# from dpufw/include/RunInstOpt.h
SHIM_DMA_BD0_BASE_ADDR = 0x1D000
SHIM_BD_OFFSET = 0x20
# from dpufw/include/dpu_func.h
DDR_AIE_ADDR_OFFSET = 0x80000000


def _get_prolog():
    return _PROLOG[:]


# based on https://github.com/Xilinx/mlir-aie/blob/cb232a43383ef3b8efd8b408545c9b74885578ad/lib/Targets/AIETargetIPU.cpp
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
def _exec_write_bd_extend_shim_tile_opt(iptr, tensor_addr=None):
    bd_id = iptr[0] & 0x0000000F
    column = (iptr[0] & 0x00FF0000) >> 16
    _addr_incr = iptr[1]
    addr_low = iptr[3]
    # upper 16 bits are for packets...
    addr_high = iptr[4] & 0x0000FFFF
    if tensor_addr is None:
        tensor_addr = (addr_high << 32) | addr_low
    tensor_addr += DDR_AIE_ADDR_OFFSET
    t_word0 = tensor_addr & 0xFFFFFFFC
    t_word1 = (iptr[4] & 0xFFFF0000) | (tensor_addr >> 32)

    base_addr = SHIM_DMA_BD0_BASE_ADDR + (bd_id * SHIM_BD_OFFSET)
    row = 0
    words = [
        *_ipu_write32(column, row, base_addr, iptr[2]),
        *_ipu_write32(column, row, base_addr + 4, t_word0),
        *_ipu_write32(column, row, base_addr + 8, t_word1),
        *_ipu_write32(column, row, base_addr + 12, iptr[5]),
        *_ipu_write32(column, row, base_addr + 16, iptr[6]),
        *_ipu_write32(column, row, base_addr + 20, iptr[7]),
        *_ipu_write32(column, row, base_addr + 24, iptr[8]),
        *_ipu_write32(column, row, base_addr + 28, iptr[9]),
    ]
    return words


# corresponds to ExecWriteBdExtendShimTileOpt
def _ipu_writebd_shimtile(
    bd_id,
    buffer_length,
    buffer_offset=0,
    ddr_id=0,
    column=0,
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
    valid_bd = 1

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
    words[3] = buffer_offset

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


class ipu:
    write32 = _ipu_write32
    shimtile_push_queue = _ipu_shimtile_push_queue
    writebd_shimtile = _ipu_writebd_shimtile
    sync = _ipu_sync
    get_prolog = _get_prolog
    _exec_write_bd_extend_shim_tile_opt = _exec_write_bd_extend_shim_tile_opt


def process_bd(
    acq_lock,
    buffer,
    rel_lock,
    *,
    offset=None,
    len=None,
    dimensions=None,
    acq_action=LockAction.AcquireGreaterEqual,
    rel_action=LockAction.Release,
    acq_val=None,
    rel_val=None,
):
    use_lock(acq_lock, acq_action, value=acq_val)
    dma_bd(buffer, offset=offset, len=len, dimensions=dimensions)
    use_lock(rel_lock, rel_action, value=rel_val)


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
        read_in_lock = lock(
            tile,
            init=1,
            sym_name=(f"{buffer_sym_name}_read_in_lock" if buffer_sym_name else None),
        )
    if write_out_lock is None:
        write_out_lock = lock(
            tile,
            init=0,
            sym_name=(f"{buffer_sym_name}_write_out_lock" if buffer_sym_name else None),
        )

    loop = repeat_count is None

    @dma(DMAChannelDir.S2MM, s2mm_channel_idx, loop=loop, repeat_count=repeat_count)
    def dma_incoming():
        process_bd(read_in_lock, buffer, write_out_lock)

    @dma(DMAChannelDir.MM2S, mm2s_channel_idx, loop=loop, repeat_count=repeat_count)
    def dma_outgoing():
        process_bd(write_out_lock, buffer, read_in_lock)


@contextmanager
def hold_lock(acq_lock, rel_lock, *, acq_val=None, rel_val=None):
    use_lock(acq_lock, LockAction.AcquireGreaterEqual, value=acq_val)
    try:
        yield
    finally:
        use_lock(rel_lock, LockAction.Release, value=rel_val)
