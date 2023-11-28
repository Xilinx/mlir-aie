# ./python/aie/dialects/aiex/__init__.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._AIEX_ops_gen import *
from .._mlir_libs._aie import *
from ..ir import *
from ..dialects import arith

# Comes from _aie
register_dialect(get_dialect_registry())


class IpuSync(IpuSyncOp):
    """Rename IpuSyncOp class"""

    def __init__(self, column, row, direction, channel, column_num=1, row_num=1):
        super().__init__(
            column=column,
            row=row,
            direction=direction,
            channel=channel,
            column_num=column_num,
            row_num=row_num,
        )


class IpuWrite32(IpuWrite32Op):
    """Specialize IpuWrite32Op class constructor to take python integers"""

    def __init__(self, column, row, address, value):
        uiTy = IntegerType.get_unsigned(32)
        intAddr = IntegerAttr.get(uiTy, address)
        intValue = IntegerAttr.get(uiTy, value)
        super().__init__(column=column, row=row, address=intAddr, value=intValue)


class IpuWriteRTP(IpuWriteRTPOp):
    """Specialize IpuWriteRTPOp class constructor to take python integers"""

    def __init__(self, name, col, row, index, value):
        uiTy = IntegerType.get_unsigned(32)
        intCol = IntegerAttr.get(uiTy, col)
        intRow = IntegerAttr.get(uiTy, row)
        intIndex = IntegerAttr.get(uiTy, index)
        super().__init__(
            buffer_sym_name=name, col=intCol, row=intRow, index=intIndex, value=value
        )


class IpuWriteBdShimTile(IpuWriteBdExShimTileOp):
    """Rename IpuWriteBdExShimTileOp class"""

    def __init__(
        self,
        column,
        column_num,
        ddr_id,
        bd_id,
        buffer_length,
        buffer_offset,
        enable_packet,
        out_of_order_id,
        packet_id,
        packet_type,
        d0_wrap,
        d0_stepsize,
        d1_wrap,
        d1_stepsize,
        d2_stepsize,
        iteration_current,
        iteration_wrap,
        iteration_stepsize,
        next_bd,
        use_next_bd,
        valid_bd,
        lock_rel_val,
        lock_rel_id,
        lock_acq_enable,
        lock_acq_val,
        lock_acq_id,
    ):
        super().__init__(
            column=column,
            column_num=column_num,
            ddr_id=ddr_id,
            bd_id=bd_id,
            buffer_length=buffer_length,
            buffer_offset=buffer_offset,
            enable_packet=enable_packet,
            out_of_order_id=out_of_order_id,
            packet_id=packet_id,
            packet_type=packet_type,
            d0_wrap=d0_wrap,
            d0_stepsize=d0_stepsize,
            d1_wrap=d1_wrap,
            d1_stepsize=d1_stepsize,
            d2_stepsize=d2_stepsize,
            iteration_current=iteration_current,
            iteration_wrap=iteration_wrap,
            iteration_stepsize=iteration_stepsize,
            next_bd=next_bd,
            use_next_bd=use_next_bd,
            valid_bd=valid_bd,
            lock_rel_val=lock_rel_val,
            lock_rel_id=lock_rel_id,
            lock_acq_enable=lock_acq_enable,
            lock_acq_val=lock_acq_val,
            lock_acq_id=lock_acq_id,
        )


class IpuDmaMemcpyNd(IpuDmaMemcpyNdOp):
    """Specialize IpuDmaMemcpyNdOp class constructor to take python integers"""

    def __init__(
        self, metadata, bd_id, mem, offsets=[0] * 4, lengths=[0] * 4, strides=[0] * 3
    ):
        symMetadata = FlatSymbolRefAttr.get(metadata)
        iTy = IntegerType.get_signless(32)
        x = 0
        y = 0
        intX = arith.ConstantOp(iTy, IntegerAttr.get(iTy, x))
        intY = arith.ConstantOp(iTy, IntegerAttr.get(iTy, y))
        valueOffsets = []
        valueLengths = []
        valueStrides = []
        for i in offsets:
            valueOffsets.append(arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)))
        for i in lengths:
            valueLengths.append(arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)))
        for i in strides:
            valueStrides.append(arith.ConstantOp(iTy, IntegerAttr.get(iTy, i)))
        super().__init__(
            metadata=symMetadata,
            id=bd_id,
            x=intX,
            y=intY,
            memref=mem,
            offset3=valueOffsets[0],
            offset2=valueOffsets[1],
            offset1=valueOffsets[2],
            offset0=valueOffsets[3],
            length3=valueLengths[0],
            length2=valueLengths[1],
            length1=valueLengths[2],
            length0=valueLengths[3],
            stride3=valueStrides[0],
            stride2=valueStrides[1],
            stride1=valueStrides[2],
        )
