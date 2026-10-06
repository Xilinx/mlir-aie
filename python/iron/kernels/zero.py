# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent, target-native zero-fill kernel."""

import math
import operator

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    TensorLayout,
    Trace,
    _arch_traits,
    _kernel_source,
    _make_extern,
    dtypes,
)

_TYPES = {
    np.int8: "int8_t",
    np.uint8: "uint8_t",
    np.int16: "int16_t",
    np.uint16: "uint16_t",
    np.int32: "int32_t",
    np.uint32: "uint32_t",
    np.float32: "float",
    bfloat16: "bfloat16",
}


@dtypes(tuple({"dtype": dtype} for dtype in _TYPES) + ({"dtype": v8bfp16ebs8},))
def zero(
    tile_size: int | tuple[int, ...] = 1024,
    dtype: type | np.dtype = np.int32,
    *,
    vectorized: bool = True,
    use_chess: bool = False,
) -> ExternalFunction:
    """Fill one tile with zeros, independently of any compute kernel.

    ``tile_size`` is an element count or shape. For ``v8bfp16ebs8`` it
    counts eight-value blocks, matching the ndarray ABI; all nine bytes
    of every block (exponent and mantissas) are cleared. Vector stores
    use the target's native width, with a scalar tail for smaller tiles.
    """
    try:
        shape = (
            (tile_size,)
            if isinstance(tile_size, (int, np.integer))
            else tuple(tile_size)
        )
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise ValueError("zero: tile_size must be a positive integer or shape") from exc
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("zero: tile_size must be a positive integer or shape")
    size = math.prod(shape)
    block = dtype is v8bfp16ebs8
    if block:
        if not _arch_traits().bfp16:
            raise NotImplementedError("zero: bfp16ebs8 requires an NPU2 device")
        from aie.utils import bfp

        layout = TensorLayout(
            (size * bfp.BLOCK,),
            pack=lambda x: bfp.encode(x).reshape(len(x), -1),
            unpack=lambda x: bfp.decode(x).reshape(len(x), -1),
        )
        reference_dtype = np.float32
        ctype, count = "uint8_t", size * bfp.BLOCK_BYTES
    else:
        dtype = np.dtype(dtype).type
        if dtype not in _TYPES:
            raise ValueError(f"zero: unsupported dtype {dtype}")
        layout = TensorLayout(shape)
        reference_dtype = dtype
        ctype, count = _TYPES[dtype], size
    flags = [f"-DZERO_TYPE={ctype}", f"-DTILE_SIZE={count}"]
    if not vectorized:
        flags.append("-DZERO_SCALAR")
    return _make_extern(
        "zero",
        _kernel_source("zero/zero.cc"),
        [np.ndarray[shape, np.dtype[dtype]]],
        compile_flags=flags,
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            layouts=(layout,),
            reference=lambda: np.zeros((1, *layout.shape), dtype=reference_dtype),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )
