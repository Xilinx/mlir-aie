# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial

from . import arith
from ._aiex_ops_gen import *
from .transform.structured import MixedValues, _dispatch_mixed_values
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
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(
            sizes
        )
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
