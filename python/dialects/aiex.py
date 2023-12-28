# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial

from . import arith
from ._aiex_ops_gen import *
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
        x = 0
        y = 0
        super().__init__(
            metadata=symMetadata,
            id=bd_id,
            x=x,
            y=y,
            memref=mem,
            offset3=offsets[0],
            offset2=offsets[1],
            offset1=offsets[2],
            offset0=offsets[3],
            length3=lengths[0],
            length2=lengths[1],
            length1=lengths[2],
            length0=lengths[3],
            stride3=strides[0],
            stride2=strides[1],
            stride1=strides[2],
        )


ipu_dma_memcpy_nd_ = ipu_dma_memcpy_nd
ipu_dma_memcpy_nd = IpuDmaMemcpyNd
