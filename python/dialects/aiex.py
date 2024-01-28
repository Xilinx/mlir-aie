# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from contextlib import contextmanager
from functools import partial

from ._aiex_ops_gen import *
from .aie import DMAChannelDir, WireBundle, LockAction, use_lock, dma_bd, dma, lock
from .transform.structured import MixedValues, _dispatch_mixed_values
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *

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

# from runtime_lib/xaiengine/aie-rt/driver/src/global/xaiemlgbl_params.h
# these aren't completely correct - right values but not necessarily the right names?
XAIEMLGBL_NOC_MODULE_DMA_MM2S_0_TASK_QUEUE = 0x0001D214
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE = 0x0001D204
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK = 0x80000000
XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK = 0x0000000F

_generated_ipu_write32 = ipu_write32


def ipu_write32(channel_dir, channel_index, col, bd_id, repeats=0):
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
    _generated_ipu_write32(address=address, column=col, row=0, value=value)


_generated_ipu_writebd_shimtile = ipu_writebd_shimtile


def ipu_writebd_shimtile(
    bd_id,
    buffer_length,
    offset,
    ddr_id,
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
    offset *= data_width // 8
    # None means do not wrap which is 0 on the arch
    if d1_size is None:
        d1_size = 0
    # None means do not wrap which is 0 on the arch
    if d0_size is None:
        d0_size = 0

    return _generated_ipu_writebd_shimtile(
        bd_id=bd_id,
        buffer_length=buffer_length,
        buffer_offset=offset,
        column=0,
        column_num=1,
        d0_size=d0_size,
        d0_stride=d0_stride,
        d1_size=d1_size,
        d1_stride=d1_stride,
        d2_stride=d2_stride,
        ddr_id=ddr_id,
        enable_packet=0,
        iteration_current=iteration_current,
        iteration_size=iteration_size,
        iteration_stride=iteration_stride,
        lock_acq_enable=lock_acq_enable,
        lock_acq_id=lock_acq_id,
        lock_acq_val=lock_acq_val,
        lock_rel_id=lock_rel_id,
        lock_rel_val=lock_rel_val,
        next_bd=next_bd,
        out_of_order_id=0,
        packet_id=0,
        packet_type=0,
        use_next_bd=use_next_bd,
        valid_bd=1,
    )


def process_bd(
    acq_lock,
    buffer,
    rel_lock,
    acq_action=LockAction.AcquireGreaterEqual,
    rel_action=LockAction.Release,
    acq_val=None,
    rel_val=None,
):
    use_lock(acq_lock, acq_action, value=acq_val)
    dma_bd(buffer)
    use_lock(rel_lock, rel_action, value=rel_val)


def forward_bd(tile, channel_idx, buffer, read_in_lock=None, write_out_lock=None):
    if read_in_lock is None:
        read_in_lock = lock(tile, init=1)
    if write_out_lock is None:
        write_out_lock = lock(tile, init=0)

    @dma(DMAChannelDir.S2MM, channel_idx)
    def dma_incoming():
        process_bd(read_in_lock, buffer, write_out_lock)

    @dma(DMAChannelDir.MM2S, channel_idx)
    def dma_outgoing():
        process_bd(write_out_lock, buffer, read_in_lock)


@contextmanager
def hold_lock(acq_lock, rel_lock):
    use_lock(acq_lock, LockAction.AcquireGreaterEqual)
    try:
        yield
    finally:
        use_lock(rel_lock, LockAction.Release)
