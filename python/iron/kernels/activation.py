# kernels/activation.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Activation kernel factories and NumPy reference implementations."""

from pathlib import Path
from typing import Callable

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    _bf16_lanes,
    _default_source_path,
    _detect_arch,
    _include_dirs,
    _kernel_source,
    _make_extern,
    _require_fixed_tile_size,
    _require_vector_alignment,
)
from .core import conv_even

_LUT_FIXED_TILE = 1024
_RUNTIME_VECTOR_WIDTH = 32


def _require_runtime_tile_size(factory_name: str, tile_size: int) -> None:
    if tile_size < _LUT_FIXED_TILE or tile_size % _RUNTIME_VECTOR_WIDTH:
        raise ValueError(
            f"{factory_name}: tile_size must be a multiple of "
            f"{_RUNTIME_VECTOR_WIDTH} and at least {_LUT_FIXED_TILE}, got {tile_size}"
        )


# Mirrors EXP_BF16_CLAMP in aie_runtime_lib/AIE2{,P}/lut_based_ops.h: the
# input domain getExpBf16 saturates to, so the references below describe what
# the device actually computes. Keep the two in step.
_EXP_BF16_CLAMP = 88.0

# The LUT-approximated kernels document rtol=0.128 (the canonical C++ testbench
# default) on their *_ref functions below; the absolute floor and the mismatch
# budget are what test/python/npu/test_kernels_e2e.py measured on device for
# tanh, sigmoid and bf16_exp.
_LUT_TOLERANCE = Tolerance.relative(
    0.128,
    0.05,
    max_mismatch_frac=0.02,
    note="LUT approximation: rtol documented on the *_ref; atol/budget from test_kernels_e2e",
)


def _unary_lut_contract(
    ref,
    *,
    count: int | None,
    tolerance: Tolerance = _LUT_TOLERANCE,
    setup: Callable[[], object] | None = conv_even,
) -> KernelContract:
    """Contract for a one-in/one-out LUT kernel, with or without a trailing count.

    The LUT kernels store bf16 from wider vector math without setting the
    core's rounding mode, so they are judged in ``conv_even``, the mode
    numpy's reference rounds in.
    """
    return KernelContract(
        roles=(In, Out, Param) if count else (In, Out),
        parameter_bindings=((2, count),) if count else (),
        reference=ref,
        tolerance=tolerance,
        acc_dtype=bfloat16,  # bf16 vector math around the LUT
        setup=setup,
        uses_lut=True,
    )


def _create_lut_kernel(
    func_name: str,
    kernel_filename: str,
    arg_types: list,
    compile_flags: list[str] | None = None,
    contract: KernelContract | None = None,
) -> ExternalFunction:
    """Create an ExternalFunction for a LUT-dependent kernel.

    Handles the aie2/aie2p split:
    - aie2: selects a kernel in the LUT-linking source translation unit.
    - aie2p: uses source_file directly (no LUT object linkage).

    ``contract`` is attached as ``.contract`` like ``_make_extern`` does.
    """
    arch = _detect_arch()
    kernel_path = _kernel_source(arch, arch, kernel_filename)

    from aie.utils import config

    include = _include_dirs()
    kernel_arch_dir = Path(config.cxx_header_path()) / "aie_kernels" / arch
    include.append(str(kernel_arch_dir))
    runtime_dir = Path(config.aie_runtime_lib_dir()) / arch.upper()
    include.append(str(runtime_dir))

    flags = list(compile_flags or [])

    if arch == "aie2":
        flags.append(f'-DAIE_LUT_KERNEL_SOURCE="{kernel_path}"')
        kernel_path = _kernel_source(arch, arch, "lut_kernel.cc")
    if compile_flags:
        return _make_extern(
            func_name,
            kernel_path,
            arg_types,
            compile_flags=flags + [f"-I{directory}" for directory in include],
            contract=contract,
        )
    return ExternalFunction(
        func_name,
        source_file=str(kernel_path),
        arg_types=arg_types,
        include_dirs=include,
        compile_flags=flags,
        contract=contract,
    )


def _bf16_lut_factory(
    factory_name: str,
    func_name: str,
    kernel_filename: str,
    tile_size: int,
    arg_arity: int,
    contract: KernelContract | None = None,
) -> ExternalFunction:
    """Build a LUT-backed bf16 kernel whose arg list is N copies of the same tile type."""
    _require_fixed_tile_size(factory_name, tile_size, _LUT_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        func_name,
        kernel_filename,
        [tile_ty] * arg_arity,
        compile_flags=(
            [f"-D{factory_name.upper()}_ELEMS={tile_size}"]
            if factory_name in ("gelu", "silu", "swiglu")
            else None
        ),
        contract=contract,
    )


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
        contract=_unary_lut_contract(
            lambda x: softmax_ref(x, tile_size=tile_size),
            count=tile_size,
            # Softmax outputs sum to 1 over the tile, so a typical element is
            # about 1 / tile_size and the generic LUT floor of 0.05 would
            # accept an all-zero output. A tenth of an average element still
            # covers the exp LUT's underflow on the far tail, while an
            # unwritten tile mismatches on most elements.
            tolerance=Tolerance.relative(
                0.128,
                0.1 / tile_size,
                max_mismatch_frac=0.02,
                note="LUT rtol from softmax_ref; atol = 0.1 / tile_size so an "
                "unwritten (all-zero) tile fails, since every softmax output is "
                "below the generic LUT atol",
            ),
            # aie2p/softmax.cc sets conv_even itself; the aie2 LUT path does not.
            setup=None if _detect_arch() == "aie2p" else conv_even,
        ),
    )


def gelu(tile_size: int = 1024) -> ExternalFunction:
    """GELU activation kernel (tanh approximation) for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "gelu",
        "gelu_bf16",
        "gelu.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(gelu_ref, count=False),
    )


def silu(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) activation kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "silu",
        "silu_bf16",
        "silu.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(silu_ref, count=False),
    )


def silu_sized(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) for bf16 tiles, with a compiled-in element count.

    Runtime-size sibling of [`silu`][iron.kernels.activation.silu]; design
    keeps the ``(in, out, size)`` ABI. Positive whole vectors are required
    (16 on aie2, 32 on aie2p).
    """
    width = _bf16_lanes()
    _require_vector_alignment("silu_sized", tile_size, width)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "silu_bf16_size",
        "silu.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DSILU_ELEMS={tile_size}"],
        contract=_unary_lut_contract(silu_ref, count=tile_size),
    )


def gelu_sized(tile_size: int = 1024) -> ExternalFunction:
    """GELU (tanh approx) for bf16 tiles, with a compiled-in element count.

    Runtime-size sibling of [`gelu`][iron.kernels.activation.gelu]; design
    keeps the ``(in, out, size)`` ABI. Positive whole vectors only: multiples
    of 16 on aie2 or 32 on aie2p.
    """
    if _detect_arch() == "aie2":
        _require_vector_alignment("gelu_sized", tile_size, 16)
    else:
        _require_vector_alignment("gelu_sized", tile_size, 32)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "gelu_bf16_size",
        "gelu.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DGELU_ELEMS={tile_size}"],
        contract=_unary_lut_contract(gelu_ref, count=tile_size),
    )


def swiglu(tile_size: int = 1024) -> ExternalFunction:
    """SwiGLU gated activation kernel for bf16 tiles (must be 1024).

    ``out = (x * w1) * silu(x * w2)``; see [`swiglu_ref`][iron.kernels.activation.swiglu_ref].
    """
    return _bf16_lut_factory(
        "swiglu",
        "swiglu_bf16",
        "swiglu.cc",
        tile_size,
        arg_arity=4,
        contract=KernelContract(
            setup=conv_even,
            roles=(In, In, In, Out),
            reference=swiglu_ref,
            acc_dtype=bfloat16,
            tolerance=_LUT_TOLERANCE,
            ops_per_call=6 * tile_size,
            uses_lut=True,
        ),
    )


def bf16_exp(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise exponential kernel for bf16 tiles (must be 1024).

    Computes ``exp(clip(x, -88, 88))``: the kernel saturates rather than
    overflowing for real inputs, including infinities. On AIE2P a
    range-reduced polynomial and integer exponent reconstruction replace
    the hardware exp2 approximation, preserving subnormal outputs. See
    [`bf16_exp_ref`][iron.kernels.activation.bf16_exp_ref] for why that
    clamp matches the AIE2 table's domain.
    """
    return _bf16_lut_factory(
        "bf16_exp",
        "exp_bf16_1024",
        "bf16_exp.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(bf16_exp_ref, count=False),
    )


def exp2f_vec(tile_size: int = 1024, min_x: float = -111.0) -> ExternalFunction:
    """Software f32 ``2**x`` kernel: a degree-5 minimax poly, not a LUT.

    A float32-output alternative to [`bf16_exp`]
    [iron.kernels.activation.bf16_exp], sharing its AIE2P range-reduced
    polynomial but with a separately configurable input domain. See
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
        contract=KernelContract(
            setup=conv_even,
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=lambda x: exp2f_vec_ref(x, min_x=min_x),
            acc_dtype=np.float32,
            tolerance=Tolerance.relative(
                1e-3,
                note="minimax poly targets 8.9e-5 relative error; see exp2f_vec_ref",
            ),
        ),
    )


def tanh(tile_size: int = 1024) -> ExternalFunction:
    """Tanh for bf16 tiles of at least 1024 elements, in multiples of 32.

    The count is compiled in; retain
    ``tile_size`` as a trailing ``int`` argument (e.g. via
    ``transform_parallel(pass_size_to_kernel=True)``).
    """
    _require_runtime_tile_size("tanh", tile_size)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "tanh_bf16",
        "tanh.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DTANH_ELEMS={tile_size}"],
        contract=_unary_lut_contract(tanh_ref, count=tile_size),
    )


def sigmoid(tile_size: int = 1024) -> ExternalFunction:
    """Sigmoid for bf16 tiles of at least 1024 elements, in multiples of 32.

    The count is compiled in; retain ``tile_size`` as a trailing ABI argument.
    """
    _require_runtime_tile_size("sigmoid", tile_size)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "sigmoid_bf16",
        "sigmoid.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DSIGMOID_ELEMS={tile_size}"],
        contract=_unary_lut_contract(sigmoid_ref, count=tile_size),
    )


def leaky_relu(tile_size: int = 1024) -> ExternalFunction:
    """Leaky ReLU for bf16 tiles of at least 64 elements, in multiples of 32.

    The count is compiled in, but the ABI retains ``(tile_size, alpha)`` as
    trailing ``int``/``bfloat16`` arguments. The slope remains runtime-valued.
    """
    if tile_size < 64 or tile_size % _RUNTIME_VECTOR_WIDTH:
        raise ValueError(
            "leaky_relu: tile_size must be a multiple of "
            f"{_RUNTIME_VECTOR_WIDTH} and at least 64, got {tile_size}"
        )
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "leaky_relu_bf16",
        "leaky_relu.cc",
        [tile_ty, tile_ty, np.int32, bfloat16],
        compile_flags=[f"-DLEAKY_RELU_ELEMS={tile_size}"],
        contract=KernelContract(
            setup=conv_even,
            roles=(In, Out, Param, Param),
            parameter_bindings=((2, tile_size),),
            reference=leaky_relu_ref,
            acc_dtype=bfloat16,
            tolerance=Tolerance.relative(
                0.03,
                0.05,
                max_mismatch_frac=0.02,
                note="exact up to bf16 rounding of alpha*x; measured by test_kernels_e2e",
            ),
        ),
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


def swiglu_ref(x, w1, w2):
    """Numpy reference for [`swiglu`][iron.kernels.activation.swiglu]: ``(x * w1) * silu(x * w2)``.

    ``swiglu.cc`` forms the two products in bf16, then ``silu`` of the second
    through the tanh LUT (``0.5 * (1 + tanh(z / 2))``). The reference rounds
    the two products to bf16 as the kernel does and computes the rest in
    float32; LUT-approximation territory, pair with ``rtol=0.128``.
    """
    xf = x.astype(np.float32)
    xw1 = (xf * w1.astype(np.float32)).astype(bfloat16).astype(np.float32)
    xw2 = (xf * w2.astype(np.float32)).astype(bfloat16).astype(np.float32)
    return (xw1 * (xw2 / (1.0 + np.exp(-xw2)))).astype(x.dtype)


def bf16_exp_ref(x):
    """Numpy reference for [`bf16_exp`][iron.kernels.activation.bf16_exp] — element-wise ``exp(x)``.

    LUT approximation territory; pair with the canonical 12.8% relative
    tolerance and ``stop_at_nonfinite=True`` (the default in
    `count_mismatches`) when verifying.

    ``exp(clip(x, -88, 88))``, not plain ``exp(x)``: the kernel clamps to
    ``EXP_BF16_CLAMP`` before its Q8 fixed-point table lookup (see
    ``aie_runtime_lib/AIE2/lut_based_ops.h``), so it saturates rather than
    overflowing. ``+88`` is the largest value the tables carry.
    ``exp(-88)`` is a nonzero bf16 subnormal (about ``6.06e-39``), not
    zero: AIE2P preserves it through integer exponent reconstruction,
    whereas the AIE2 LUT may flush the tail under the absolute tolerance.
    The clamp also keeps the reference itself in range:
    ``exp(88) = 1.65e+38`` fits float32 where ``exp(89)`` would not.
    """
    xf = np.clip(x.astype(np.float32), -_EXP_BF16_CLAMP, _EXP_BF16_CLAMP)
    # The clamp rules out overflow, so that warning stays un-suppressed and
    # would now be a real signal. A NaN input still reaches exp -- the kernel
    # declares nonfinite="unspecified" and callers do feed raw bit patterns
    # (programming_examples/basic/vector_exp sweeps all 65536 of them).
    with np.errstate(invalid="ignore"):
        return np.exp(xf).astype(x.dtype)


def exp2f_vec_ref(x, min_x: float = -111.0):
    """Numpy reference for [`exp2f_vec`][iron.kernels.activation.exp2f_vec]: exact ``2**x``.

    Unlike the LUT-based refs above, this is float64 ``2**x`` (not a
    reimplementation of the on-device poly): the kernel targets ~8.9e-5
    relative error by design, several orders tighter than the LUT-based
    kernels' 12.8% default, so pair with a correspondingly tight
    tolerance (e.g. ``rtol=1e-3``) rather than the LUT default.

    The kernel clamps its input to ``min_x`` before evaluating (see the
    factory's ``min_x``), so the reference does the same: ``2**-5000`` is
    ``2**min_x`` on the device, not zero. Pass the factory's ``min_x``.
    """
    xf = np.maximum(x.astype(np.float64), min_x)
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
