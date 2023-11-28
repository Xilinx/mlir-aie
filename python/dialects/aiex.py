# ./python/aie/dialects/aiex/__init__.py -*- Python -*-
from functools import partial

from . import arith
from ._AIEX_ops_gen import *
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from ..ir import FlatSymbolRefAttr, IntegerType, IntegerAttr

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Comes from _aie
register_dialect(get_dialect_registry())

ipu_sync = partial(ipu_sync, column_num=1, row_num=1)


class IpuDmaMemcpyNd(IpuDmaMemcpyNdOp):
    """Specialize IpuDmaMemcpyNdOp class constructor to take python integers"""

    def __init__(self, metadata, bd_id, mem, offsets=None, lengths=None, strides=None):
        if strides is None:
            strides = [0] * 3
        if lengths is None:
            lengths = [0] * 4
        if offsets is None:
            offsets = [0] * 4
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


ipu_dma_memcpy_nd = IpuDmaMemcpyNd
