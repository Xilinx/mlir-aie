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
    Param,
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
    window: int | None = None,
    use_chess: bool = False,
) -> ExternalFunction:
    """Fill one tile with zeros, independently of any compute kernel.

    ``tile_size`` is an element count or shape. For ``v8bfp16ebs8`` it
    counts eight-value blocks, matching the ndarray ABI; all nine bytes
    of every block (exponent and mantissas) are cleared. Vector stores
    use the target's native width, with a scalar tail for smaller tiles.

    With ``window``, each call zeroes ``window`` elements of the one tile
    at a runtime offset, ``call * window``, so the calls fill it between
    them from starts of every alignment.
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
    arg_types = [np.ndarray[shape, np.dtype[dtype]]]
    roles, layouts, bindings, out_offset = (Out,), (layout,), (), None
    if window is not None:
        if block or not vectorized:
            raise ValueError("zero: window needs a plain dtype and vectorized=True")
        if window <= 0 or size % window:
            raise ValueError(f"zero: window {window} must divide tile_size {size}")
        arg_types += [np.int32, np.int32]
        layout = TensorLayout((window,))
        roles, layouts = (Out, Param, Param), (layout, None, None)
        bindings, out_offset = ((1, 0), (2, window)), (1, window)
    return _make_extern(
        "zero" if window is None else "zero_window",
        _kernel_source("zero/zero.cc"),
        arg_types,
        compile_flags=flags,
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=roles,
            layouts=layouts,
            parameter_bindings=bindings,
            out_offset=out_offset,
            reference=lambda: np.zeros((1, *layout.shape), dtype=reference_dtype),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )
