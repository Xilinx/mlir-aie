# kernels/activation.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Activation kernel factories + numpy reference implementations.

Factories (each returns an [`ExternalFunction`][iron.ExternalFunction]):
  softmax, gelu, silu, swiglu, bf16_exp, exp2f_vec, tanh, sigmoid, leaky_relu.

The ones whose element count is a runtime argument -- tanh, sigmoid,
leaky_relu, and the ``*_sized`` siblings -- take any tile their inner loop
can step through, not just 1024; ``_require_runtime_tile_size`` says what
that means.

Companion numpy reference implementations for host-side verification:
  [`relu_ref`][iron.kernels.activation.relu_ref], [`silu_ref`][iron.kernels.activation.silu_ref], [`gelu_ref`][iron.kernels.activation.gelu_ref],
  [`bf16_exp_ref`][iron.kernels.activation.bf16_exp_ref], [`softmax_ref`][iron.kernels.activation.softmax_ref],
  [`exp2f_vec_ref`][iron.kernels.activation.exp2f_vec_ref].  These compute the AIE
  kernel's op in float32 so designs don't each reimplement the math
  in their verify path.  Pair with
  `count_mismatches` (rtol=0.128 is the
  canonical LUT-tolerance default; see each ref's docstring for
  per-op recommendations).
"""

from pathlib import Path

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import (
    _default_source_path,
    _detect_arch,
    _include_dirs,
    _kernel_source,
    _make_extern,
    _require_fixed_tile_size,
)

_LUT_FIXED_TILE = 1024

# The kernels whose element count is a runtime argument, and what their inner
# loop is compiled with per architecture: (vector width, minimum trip count).
#
# The width is a hard requirement -- the loop steps by it and loads a full
# vector, so a tile that is not a multiple of it runs off the end of the
# buffer. The trip count is AIE_LOOP_MIN_ITERATION_COUNT, a promise to the
# pipeliner: advisory under Peano, which emits the low-trip guard anyway, and
# a contract under xchesscc, which may drop it. So the minimum is enforced
# only for use_chess=True, and a shorter tile is allowed on Peano.
_RUNTIME_SIZED_LOOPS = {
    "tanh_bf16": {"aie2": (32, 32), "aie2p": (32, 32)},
    "sigmoid_bf16": {"aie2": (32, 32), "aie2p": (32, 32)},
    "leaky_relu_bf16": {"aie2": (16, 4), "aie2p": (32, 2)},
}


def _require_runtime_tile_size(
    factory_name: str, func_name: str, tile_size: int, use_chess: bool
) -> None:
    """Check a runtime-sized kernel's tile against its inner loop.

    These kernels take the element count as an argument, so the tile is not
    fixed at 1024 the way a size-baked-in kernel's is; what it must satisfy
    is the loop itself. See ``_RUNTIME_SIZED_LOOPS``.
    """
    arch = _detect_arch()
    width, min_iterations = _RUNTIME_SIZED_LOOPS[func_name][arch]
    if tile_size % width:
        raise ValueError(
            f"{factory_name}() tile_size must be a multiple of {width} on "
            f"{arch} -- {func_name}'s loop steps by {width} and loads a full "
            f"vector -- got {tile_size}."
        )
    floor = width * min_iterations
    if use_chess and tile_size < floor:
        raise ValueError(
            f"{factory_name}() tile_size must be at least {floor} under "
            f"use_chess=True: {func_name} promises the pipeliner "
            f"{min_iterations} iterations of {width} elements "
            f"(AIE_LOOP_MIN_ITERATION_COUNT), which xchesscc takes as a "
            f"contract. Got {tile_size}."
        )


def _create_lut_kernel(
    func_name: str,
    kernel_filename: str,
    arg_types: list,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
) -> ExternalFunction:
    """Create an ExternalFunction for a LUT-dependent kernel.

    Handles the aie2/aie2p split:
    - aie2: combines kernel source with lut_based_ops.cpp in a single TU.
    - aie2p: uses source_file directly (no LUT dependency).
    """
    arch = _detect_arch()
    kernel_path = _kernel_source(arch, arch, kernel_filename)

    from aie.utils import config

    include = _include_dirs()
    kernel_arch_dir = Path(config.cxx_header_path()) / "aie_kernels" / arch
    include.append(str(kernel_arch_dir))

    flags = compile_flags or []

    if arch == "aie2":
        runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
        lut_cpp = runtime_dir / "lut_based_ops.cpp"
        include.append(str(runtime_dir))
        source = f'#include "{kernel_path}"\n#include "{lut_cpp}"\n'
        return ExternalFunction(
            func_name,
            source_string=source,
            arg_types=arg_types,
            include_dirs=include,
            compile_flags=flags,
            use_chess=use_chess,
        )
    return ExternalFunction(
        func_name,
        source_file=str(kernel_path),
        arg_types=arg_types,
        include_dirs=include,
        compile_flags=flags,
        use_chess=use_chess,
    )


def _bf16_lut_factory(
    factory_name: str,
    func_name: str,
    kernel_filename: str,
    tile_size: int,
    arg_arity: int,
) -> ExternalFunction:
    """Build a LUT-backed bf16 kernel whose arg list is N copies of the same tile type."""
    _require_fixed_tile_size(factory_name, tile_size, _LUT_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(func_name, kernel_filename, [tile_ty] * arg_arity)


def softmax(tile_size: int = 1024) -> ExternalFunction:
    """Softmax activation kernel for bf16 tiles (tile_size must be 1024).

    Args:
        tile_size: Number of elements per tile.

    Returns:
        ExternalFunction configured for the softmax kernel.
    """
    _require_fixed_tile_size("softmax", tile_size, _LUT_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "softmax_bf16",
        "softmax.cc",
        [tile_ty, tile_ty, np.int32],
    )


def gelu(tile_size: int = 1024) -> ExternalFunction:
    """GELU activation kernel (tanh approximation) for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory("gelu", "gelu_bf16", "gelu.cc", tile_size, arg_arity=2)


def silu(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) activation kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory("silu", "silu_bf16", "silu.cc", tile_size, arg_arity=2)


def silu_sized(tile_size: int = 1024, use_chess: bool = False) -> ExternalFunction:
    """SiLU (Swish) for bf16 tiles, element count read at runtime.

    Runtime-size sibling of [`silu`][iron.kernels.activation.silu]; design
    passes ``(in, out, size)``.  Any ``tile_size`` is allowed.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "silu_bf16_size", "silu.cc", [tile_ty, tile_ty, np.int32], use_chess=use_chess
    )


def gelu_sized(tile_size: int = 1024, use_chess: bool = False) -> ExternalFunction:
    """GELU (tanh approx) for bf16 tiles, element count read at runtime.

    Runtime-size sibling of [`gelu`][iron.kernels.activation.gelu]; design
    passes ``(in, out, size)``.  Any ``tile_size`` is allowed.
    """
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "gelu_bf16_size", "gelu.cc", [tile_ty, tile_ty, np.int32], use_chess=use_chess
    )


def swiglu(tile_size: int = 1024) -> ExternalFunction:
    """SwiGLU gated activation kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "swiglu", "swiglu_bf16", "swiglu.cc", tile_size, arg_arity=4
    )


def bf16_exp(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise exponential kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "bf16_exp", "exp_bf16_1024", "bf16_exp.cc", tile_size, arg_arity=2
    )


def exp2f_vec(tile_size: int = 1024, min_x: float = -111.0) -> ExternalFunction:
    """Software f32 ``2**x`` kernel: a degree-5 minimax poly, not a LUT.

    An accuracy-tradeoff alternative to the LUT-based [`bf16_exp`]
    [iron.kernels.activation.bf16_exp] path for callers (softmax, sigmoid)
    that need better than the LUT's domain-dependent error (worst on
    negative inputs, which is exactly softmax's range). See
    ``aie_kernels/aie2p/exp2f_vec.cc`` for the accuracy rationale and the
    ``noinline`` codegen hazard this kernel carries.

    aie2p only for now; not characterized on aie2.

    Args:
        tile_size: Number of elements per tile; must be a multiple of 16
            (the kernel's vector width).
        min_x: Input is clamped to this before evaluation. The default
            -111 is the lowest exponent that still holds the kernel's
            8.9e-5 relative error; -126 is the hard floor (one f32
            exponent field), reachable at up to 6.5e-3. See
            ``aie_kernels/aie2p/exp2f_vec.cc`` for the measured table.

    Returns:
        ExternalFunction configured for the exp2f_vec kernel.

    Raises:
        NotImplementedError: On aie2 (this kernel has not been ported).
        ValueError: If tile_size is not a multiple of 16, or min_x is
            below -126.
    """
    if tile_size % 16 != 0:
        raise ValueError(
            f"exp2f_vec: tile_size must be a multiple of 16, got {tile_size}"
        )
    if min_x < -126.0:
        raise ValueError(
            f"exp2f_vec: min_x must be >= -126 (the kernel builds 2**k in the "
            f"f32 exponent field, whose smallest normal exponent is -126), "
            f"got {min_x}"
        )
    arch = _detect_arch()
    if arch != "aie2p":
        raise NotImplementedError(
            "exp2f_vec is aie2p-only for now; it has not been characterized "
            "or ported to aie2"
        )
    source = _default_source_path("exp2f_vec.cc")
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    return _make_extern(
        "exp2f_vec_f32",
        source,
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DEXP2F_VEC_MIN_X={float(min_x)!r}f"],
    )


def tanh(tile_size: int = 1024, use_chess: bool = False) -> ExternalFunction:
    """Tanh activation kernel for bf16 tiles.

    The kernel takes the element count at runtime, so the design must pass
    ``tile_size`` as a trailing ``int`` argument (e.g. via
    ``transform_parallel(pass_size_to_kernel=True)``), and any multiple of
    32 is a legal tile. Under ``use_chess`` the tile must also be at least
    1024; see ``_require_runtime_tile_size``.
    """
    _require_runtime_tile_size("tanh", "tanh_bf16", tile_size, use_chess)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "tanh_bf16", "tanh.cc", [tile_ty, tile_ty, np.int32], use_chess=use_chess
    )


def sigmoid(tile_size: int = 1024, use_chess: bool = False) -> ExternalFunction:
    """Sigmoid activation kernel for bf16 tiles.

    Runtime element count — pass ``tile_size`` as a trailing ``int``
    argument. Any multiple of 32 is a legal tile; under ``use_chess`` it
    must also be at least 1024. See ``_require_runtime_tile_size``.
    """
    _require_runtime_tile_size("sigmoid", "sigmoid_bf16", tile_size, use_chess)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "sigmoid_bf16", "sigmoid.cc", [tile_ty, tile_ty, np.int32], use_chess=use_chess
    )


def leaky_relu(tile_size: int = 1024, use_chess: bool = False) -> ExternalFunction:
    """Leaky ReLU activation kernel for bf16 tiles.

    Takes the element count and the ``alpha`` slope at runtime, so the design
    must pass ``(tile_size, alpha)`` as trailing ``int``/``bfloat16``
    arguments. The tile must be a multiple of the architecture's vector
    width (16 on aie2, 32 on aie2p), and under ``use_chess`` at least 64.
    See ``_require_runtime_tile_size``.
    """
    _require_runtime_tile_size("leaky_relu", "leaky_relu_bf16", tile_size, use_chess)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "leaky_relu_bf16",
        "leaky_relu.cc",
        [tile_ty, tile_ty, np.int32, bfloat16],
        use_chess=use_chess,
    )


# ---------------------------------------------------------------------------
# Reference (numpy) implementations
# ---------------------------------------------------------------------------
# The kernels above are LUT approximations.  The functions below compute the
# corresponding op in float32 numpy so host harnesses can verify the AIE
# output without each design re-implementing the math.  Output dtype matches
# the input (so a bf16 input yields a bf16 reference, comparable to the AIE
# kernel output via ``aie.utils.verify.{nearly_equal, count_mismatches}``).


def relu_ref(x):
    """Numpy reference for a ReLU kernel — element-wise `max(x, 0)`.

    Exact; tolerance comparison is not needed.  See `aie.utils.verify`
    for the relaxed bf16/LUT-style comparators most kernels here want.
    """
    return np.maximum(x.astype(np.float32), 0.0).astype(x.dtype)


def silu_ref(x):
    """Numpy reference for [`silu`][iron.kernels.activation.silu] (Swish) — ``x * sigmoid(x)``.

    LUT-approximation territory; pair with ``rtol=0.128`` (the default
    in `count_mismatches`) when verifying.
    """
    xf = x.astype(np.float32)
    return (xf / (1.0 + np.exp(-xf))).astype(x.dtype)


def gelu_ref(x):
    """Numpy reference for [`gelu`][iron.kernels.activation.gelu].

    Tanh approximation ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
    Matches the C++ kernel's tanh-GELU formula; pair with ``rtol=0.128,
    atol=0.05`` when verifying.
    """
    import math as _math

    xf = x.astype(np.float32)
    return (
        0.5 * xf * (1.0 + np.tanh(_math.sqrt(2.0 / _math.pi) * (xf + 0.044715 * xf**3)))
    ).astype(x.dtype)


def tanh_ref(x):
    """Numpy reference for [`tanh`][iron.kernels.activation.tanh] — element-wise ``tanh(x)``.

    LUT/native-approximation territory; pair with ``rtol=0.128`` when verifying.
    """
    return np.tanh(x.astype(np.float32)).astype(x.dtype)


def sigmoid_ref(x):
    """Numpy reference for [`sigmoid`][iron.kernels.activation.sigmoid] — ``1 / (1 + exp(-x))``.

    LUT-approximation territory; pair with ``rtol=0.128`` when verifying.
    """
    xf = x.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-xf))).astype(x.dtype)


def leaky_relu_ref(x, alpha=0.01):
    """Numpy reference for [`leaky_relu`][iron.kernels.activation.leaky_relu].

    ``x if x > 0 else alpha * x``.  ``alpha`` must match the slope the design
    passes to the kernel at runtime.  Exact up to bf16 rounding; pair with a
    small ``rtol`` when verifying.
    """
    xf = x.astype(np.float32)
    return np.where(xf > 0.0, xf, alpha * xf).astype(x.dtype)


def bf16_exp_ref(x):
    """Numpy reference for [`bf16_exp`][iron.kernels.activation.bf16_exp] — element-wise ``exp(x)``.

    LUT approximation territory; the AIE kernel saturates on large inputs.
    Pair with the canonical 12.8% relative tolerance and ``stop_at_
    nonfinite=True`` (the default in
    `count_mismatches`) when verifying.
    """
    xf = x.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(xf).astype(x.dtype)


def exp2f_vec_ref(x):
    """Numpy reference for [`exp2f_vec`][iron.kernels.activation.exp2f_vec]: exact ``2**x``.

    Unlike the LUT-based refs above, this is float64 ``2**x`` (not a
    reimplementation of the on-device poly): the kernel targets ~8.9e-5
    relative error by design, several orders tighter than the LUT-based
    kernels' 12.8% default, so pair with a correspondingly tight
    tolerance (e.g. ``rtol=1e-3``) rather than the LUT default.
    """
    xf = x.astype(np.float64)
    return np.exp2(xf).astype(x.dtype)


def softmax_ref(x, *, tile_size: int = 1024):
    """Numpy reference for [`softmax`][iron.kernels.activation.softmax].

    The AIE kernel computes softmax independently per ``tile_size``-element
    tile (no cross-tile reduction), so the reference splits ``x`` the same
    way before applying the float32 softmax.  ``x.size`` must be a
    multiple of ``tile_size``.
    """
    xf = x.astype(np.float32)
    if xf.size % tile_size != 0:
        raise ValueError(
            f"softmax_ref: x has {xf.size} elements; not a multiple of "
            f"tile_size={tile_size}"
        )
    flat = xf.reshape(-1, tile_size)
    flat = flat - flat.max(axis=1, keepdims=True)
    exp = np.exp(flat)
    out = exp / exp.sum(axis=1, keepdims=True)
    return out.reshape(x.shape).astype(x.dtype)
