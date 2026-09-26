# kernels/activation.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Activation kernel factories and NumPy reference implementations."""

import math
from pathlib import Path
from typing import Callable

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.accuracy import round_to
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    ARCH_TRAITS,
    KernelContract,
    Param,
    Trace,
    _arch_traits,
    _detect_arch,
    _include_dirs,
    _kernel_source,
    _make_extern,
    _require_fixed_tile_size,
    _require_vector_alignment,
    _tuned_arch,
)
from .core import conv_even

_LUT_FIXED_TILE = 1024
_RUNTIME_VECTOR_WIDTH = 32


# Mirrors EXP_BF16_CLAMP in aie_runtime_lib/AIE2{,P}/lut_based_ops.h: the
# input domain getExpBf16 saturates to, so the references below describe what
# the device actually computes. Keep the two in step.
_EXP_BF16_CLAMP = 88.0

# The 32-entry piecewise-linear tanh table of aie_runtime_lib/*/lut_based_ops.cpp,
# de-duplicated from its 4-way bank replication. getTanhBf16 computes
# slope[e] * x + offset[e] for e = clamp(floor(4x), -16, 15) + 16, i.e. 0.25-wide
# segments over [-4, 4) saturating to -1 / +1 outside. Modelling it exactly is
# what lets the LUT build be judged at one bf16 ulp instead of a percentage.
_TANH_LUT_SLOPE = (
    0.0,
    0.002838134765625,
    0.005096435546875,
    0.00750732421875,
    0.0126953125,
    0.021240234375,
    0.035400390625,
    0.056396484375,
    0.091796875,
    0.1455078125,
    0.2294921875,
    0.34765625,
    0.50390625,
    0.69140625,
    0.8671875,
    1.0,
    1.0,
    0.8671875,
    0.69140625,
    0.50390625,
    0.34765625,
    0.2294921875,
    0.1455078125,
    0.091796875,
    0.056396484375,
    0.035400390625,
    0.021240234375,
    0.0126953125,
    0.00750732421875,
    0.005096435546875,
    0.002838134765625,
    0.0,
)
_TANH_LUT_OFFSET = (
    -1.0,
    -0.98828125,
    -0.98046875,
    -0.97265625,
    -0.95703125,
    -0.93359375,
    -0.8984375,
    -0.8515625,
    -0.78125,
    -0.6875,
    -0.5625,
    -0.416015625,
    -0.259765625,
    -0.11962890625,
    -0.03076171875,
    0.0,
    0.0,
    0.03076171875,
    0.11962890625,
    0.259765625,
    0.416015625,
    0.5625,
    0.6875,
    0.78125,
    0.8515625,
    0.8984375,
    0.93359375,
    0.95703125,
    0.97265625,
    0.98046875,
    0.98828125,
    1.0,
)

# The LUT build is judged against the model above, which reproduces the
# kernel's arithmetic step for step; only the final accumulator-to-bf16 store
# can differ, and only by a rounding.
_LUT_MODEL_TOLERANCE = Tolerance.bf16_ulps(
    1,
    note="exact model of getTanhBf16; measured bit-exact on npu2 over 4096 "
    "values, one ulp left for the accfloat->bf16 store's rounding mode",
)

# vtanh has no published spec, so a model reverse-engineered from the device
# would pass by construction. It keeps the true-function reference instead,
# bounded by what vtanh costs: worst at x = 0.5, where it still returns its
# argument.
_VTANH_TOLERANCE = Tolerance.relative(
    0.05,
    0.001,
    note="AIE2P vtanh approximation, measured on npu2: 3.79e-2 abs worst at "
    "x=0.5 (0.0394 rel). Use tanh(use_lut=True) for 7.5x tighter",
)


# Kernels that reach vtanh through the sigmoid identity (1+tanh(x/2))/2, which
# roughly doubles vtanh's 0.0392 relative to 0.0787: it halves the absolute
# error but divides by a value that shrinks faster. silu and swiglu inherit
# that figure, their further steps being bf16 multiplies. swiglu's output
# xw1 * silu(xw2) is the one that goes to zero while its error does not.
_VTANH_FAMILY_BOUNDS = {
    "sigmoid": (0.08, 0.025),  # measured 0.0787 rel, 1.95e-2 abs (output <= 1)
    "silu": (0.08, 0.035),  # measured 0.0784 rel, 3.12e-2 abs
    "swiglu": (0.08, 0.350),  # measured 0.25 abs at a near-zero expected value
}


# aie2p's bf16_exp.cc evaluates a range-reduced polynomial rather than reading
# getExpBf16's tables, so the LUT model does not describe it.
_EXP_POLY_TOLERANCE = Tolerance.relative(
    0.005,
    1e-38,
    note="AIE2P range-reduced polynomial, measured on npu2; aie2 uses the "
    "LUT and is judged against bf16_exp_lut_ref instead",
)
_EXP_LUT_TOLERANCE = Tolerance.bf16_ulps(
    1,
    atol=2.0**-126,
    note="exact getExpBf16 model with one store ulp; atol admits AIE2's "
    "subnormal flush to zero",
)


# gelu's tanh approximation is its own, and split per architecture; it is not
# the sigmoid-identity family above. gelu(x) goes to zero for negative x, so
# this is one of the bounds that needs the floor.
_GELU_TOLERANCE = Tolerance.relative(
    0.05,
    0.020,
    note="gelu tanh approximation, measured on npu2: 1.56e-2 abs at a "
    "near-zero expected value",
)


# What AIE2P's vtanh returns, measured on npu2 through gelu over every finite
# bf16 input and through fused_mm's sigmoid over 262144 f32 arguments: u
# itself up to |u| = 0.5, then a ramp that meets tanh by 0.8, and exactly
# +/-1 from |u| = 3 on. Between, it is piecewise with breaks every 0.25:
# within 1.9 ulps of tanh (2.5 allowed), except the piece from 1 to 1.25,
# which starts 4.03 ulps off (4.5 allowed).
_VTANH_ARG_BAND = (0.5, 0.8)
_VTANH_ULPS = 2.5
_VTANH_WORST_PIECE = (1.0, 1.25, 4.5)
_VTANH_SATURATES = 3.0


def _bf16_ulp(v):
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(v), 2.0**-126))) - 7)


def _vtanh_error(u):
    """Bound on ``|vtanh(u) - tanh(u)|``, elementwise."""
    a = np.abs(u)
    t = np.tanh(a)
    lo, hi = _VTANH_ARG_BAND
    band = np.where(
        a <= lo,
        a - t,
        (lo - math.tanh(lo)) * np.clip((hi - a) / (hi - lo), 0.0, 1.0),
    )
    start, end, worst = _VTANH_WORST_PIECE
    ulps = np.where((a >= start) & (a < end), worst, _VTANH_ULPS)
    return np.where(a >= _VTANH_SATURATES, 1.0 - t, band + ulps * _bf16_ulp(t))


def _gelu_vtanh_bound(x):
    """Bound on AIE2P gelu.cc's error at ``x``, against bf16 of the true gelu.

    The kernel's tanh argument ``u`` is off by sqrt(2/pi) stored in bf16 and
    by the bf16 roundings of ``x*x`` and ``x*s_beta``; tanh's slope
    ``1 - t*t`` carries that through. vtanh adds its own error, and
    ``x/2 * (1 + t)`` scales the sum by ``|x|/2``. One output ulp covers the
    final store, and the smallest normal covers a subnormal flushed to zero.
    The kernel clamps -inf, so it is bounded as the most negative f32.
    """
    x = np.maximum(np.asarray(x, np.float64), -np.finfo(np.float32).max)
    s = math.sqrt(2 / math.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        cubic = 0.044715 * x**3
        u = s * (x + cubic)
        du = abs(s - float(bfloat16(s))) / s * np.abs(u) + 2.0**-7 * s * np.abs(cubic)
        t = np.tanh(u)
        et = _vtanh_error(u) + (1 - t * t) * du
        return 0.5 * np.abs(x) * et + _bf16_ulp(0.5 * x * (1 + t)) + 2.0**-126


# On AIE2P the per-input bound fails a 1.5% change to sqrt(2/pi), which no
# single rtol/atol does while passing the kernel: at x = -0.6 vtanh's
# identity band is 11 ulps off, more than the mutation moves anything.
_GELU_VTANH_TOLERANCE = Tolerance.bounded(
    _gelu_vtanh_bound,
    note="gelu.cc's bf16 roundings plus vtanh's error, measured over every "
    "finite bf16 input on npu2; fails a 1.5% change to sqrt(2/pi)",
)


def _gelu_tolerance() -> Tolerance:
    return _GELU_VTANH_TOLERANCE if _arch_traits().native_tanh else _GELU_TOLERANCE


def _vtanh_family_tolerance(name: str) -> Tolerance:
    """Return the measured vtanh bound for a kernel reaching tanh via sigmoid."""
    rtol, atol = _VTANH_FAMILY_BOUNDS[name]
    return Tolerance.relative(
        rtol,
        atol,
        note=f"AIE2P vtanh via sigmoid identity, measured on npu2 for {name}; "
        f"use {name}(use_lut=True) to be judged against an exact model instead",
    )


def _unary_lut_contract(
    ref,
    *,
    count: int | None,
    tolerance: Tolerance,
    setup: Callable[[], object] | None = conv_even,
    use_lut: bool = False,
    elementwise: Callable | None = None,
    lut_tolerance: Tolerance = _LUT_MODEL_TOLERANCE,
    stack_bytes: int | None = None,
) -> KernelContract:
    """Contract for a one-in/one-out LUT kernel, with or without a trailing count.

    ``tolerance`` is required. It used to default to a shared 12.8%-relative
    bound carrying a 2% budget of arbitrarily-wrong elements -- a C++
    testbench default that several kernels inherited without anyone deriving
    it. Every bound here is now measured or modelled per kernel, and a new
    one has to say which.

    The LUT kernels store bf16 from wider vector math without setting the
    core's rounding mode, so they are judged in ``conv_even``, the mode
    numpy's reference rounds in.

    ``use_lut`` selects the reference to match the tanh the kernel was built
    with. The LUT is a documented 32-segment interpolation this package models
    exactly, so that build is judged against the model at one bf16 ulp. vtanh
    has no published spec, so that build keeps the true-function reference and
    a tolerance sized to the instruction's measured error -- a much weaker
    statement, and deliberately a different one.
    """
    # Without a tanh instruction the LUT path is taken whatever the caller
    # asked for, and gets the exact model too.
    if (use_lut or not _arch_traits().native_tanh) and elementwise is not None:
        ref, tolerance = elementwise, lut_tolerance
    return KernelContract(
        trace=Trace.whole_call(),
        roles=(In, Out, Param) if count else (In, Out),
        parameter_bindings=((2, count),) if count else (),
        reference=ref,
        tolerance=tolerance,
        acc_dtype=bfloat16,  # bf16 vector math around the LUT
        setup=setup,
        uses_lut=True,
        stack_bytes=stack_bytes,
    )


def _create_lut_kernel(
    func_name: str,
    kernel_filename: str,
    arg_types: list,
    compile_flags: list[str] | None = None,
    contract: KernelContract | None = None,
    use_lut_tanh: bool = False,
) -> ExternalFunction:
    """Create an ExternalFunction for a LUT-dependent kernel.

    ``use_lut_tanh`` asks for getTanhBf16 over the vtanh instruction. It is
    moot on aie2, which has no tanh instruction and always reads the tables.

    A build that reads a table is compiled inside common/lut_kernel.cc, which is
    what pulls lut_based_ops.cpp -- and therefore the tables -- into the
    translation unit.

    ``contract`` is attached as ``.contract`` like ``_make_extern`` does.
    """
    arch = _detect_arch()
    kernel_path = _kernel_source(f"activation/{kernel_filename}")

    from aie.utils import config

    include = _include_dirs()
    # lut_kernel.cc lives elsewhere, so the kernel's own directory goes on the
    # include path for its relative includes.
    include.append(str(kernel_path.parent))
    runtime_dir = Path(config.aie_runtime_lib_dir()) / arch.upper()
    include.append(str(runtime_dir))

    flags = list(compile_flags or [])
    if use_lut_tanh and ARCH_TRAITS[arch].native_tanh:
        flags.append("-DACTIVATIONS_TANH_LUT=1")

    if use_lut_tanh or not ARCH_TRAITS[arch].native_tanh:
        flags.append(f'-DAIE_LUT_KERNEL_SOURCE="{kernel_path}"')
        kernel_path = _kernel_source("common/lut_kernel.cc")
    return _make_extern(
        func_name,
        kernel_path,
        arg_types,
        compile_flags=flags,
        contract=contract,
        include_dirs=include,
    )


def _bf16_lut_factory(
    factory_name: str,
    func_name: str,
    kernel_filename: str,
    tile_size: int,
    arg_arity: int,
    contract: KernelContract | None = None,
    use_lut_tanh: bool = False,
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
        use_lut_tanh=use_lut_tanh,
    )


def softmax(tile_size: int = 1024) -> ExternalFunction:
    """Softmax activation kernel for bf16 tiles.

    Args:
        tile_size: Number of elements per tile, passed at run time; a
            positive multiple of 32, the kernel's vector step.

    Returns:
        ExternalFunction configured for the softmax kernel.
    """
    _require_vector_alignment("softmax", tile_size, _RUNTIME_VECTOR_WIDTH)
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
                0.04,
                0.1 / tile_size,
                note="AIE2P exp instruction through the softmax normalisation, "
                "measured on npu2 at 2.94e-2 relative; atol = 0.1 / tile_size "
                "so an unwritten (all-zero) tile fails, since every softmax "
                "output is below a generic absolute floor",
            ),
            # softmax_aie2p.h sets conv_even itself; the aie2 LUT path does not.
            setup=conv_even if _tuned_arch() == "aie2" else None,
        ),
    )


# aiecc measured 1120 on aie2, past the 1 KiB default.
_GELU_AIE2_STACK_BYTES = 1280


def gelu(tile_size: int = 1024) -> ExternalFunction:
    """GELU activation kernel (tanh approximation) for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "gelu",
        "gelu_bf16",
        "gelu.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(
            gelu_ref,
            count=False,
            tolerance=_gelu_tolerance(),
            stack_bytes=_GELU_AIE2_STACK_BYTES if _tuned_arch() == "aie2" else None,
        ),
    )


def silu(tile_size: int = 1024, use_lut: bool = False) -> ExternalFunction:
    """SiLU (Swish) activation kernel for bf16 tiles (must be 1024).

    Args:
        tile_size: Elements per call (must be 1024).
        use_lut: Compute tanh from the interpolated LUT rather than AIE2P's
            vtanh instruction, which makes this 7.5x closer to the true
            function and judged against an exact model of it. Moot on aie2.
    """
    return _bf16_lut_factory(
        "silu",
        "silu_bf16",
        "silu.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(
            silu_ref,
            count=False,
            use_lut=use_lut,
            elementwise=silu_table_ref if _tuned_arch() == "aie2p" else silu_lut_ref,
            tolerance=_vtanh_family_tolerance("silu"),
        ),
        use_lut_tanh=use_lut,
    )


def silu_sized(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) for bf16 tiles, with a compiled-in element count.

    Runtime-size sibling of [`silu`][iron.kernels.activation.silu]; design
    keeps the ``(in, out, size)`` ABI. Positive whole vectors are required
    (16 on aie2, 32 on aie2p).
    """
    width = _arch_traits().bf16_lanes
    _require_vector_alignment("silu_sized", tile_size, width)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "silu_bf16_size",
        "silu.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DSILU_ELEMS={tile_size}"],
        contract=_unary_lut_contract(
            silu_ref, count=tile_size, tolerance=_vtanh_family_tolerance("silu")
        ),
    )


def gelu_sized(tile_size: int = 1024) -> ExternalFunction:
    """GELU (tanh approx) for bf16 tiles, with a compiled-in element count.

    Runtime-size sibling of [`gelu`][iron.kernels.activation.gelu]; design
    keeps the ``(in, out, size)`` ABI. Positive whole vectors only: multiples
    of 16 on aie2 or 32 on aie2p.
    """
    _require_vector_alignment("gelu_sized", tile_size, _arch_traits().bf16_lanes)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "gelu_bf16_size",
        "gelu.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DGELU_ELEMS={tile_size}"],
        contract=_unary_lut_contract(
            gelu_ref,
            count=tile_size,
            tolerance=_gelu_tolerance(),
            stack_bytes=_GELU_AIE2_STACK_BYTES if _tuned_arch() == "aie2" else None,
        ),
    )


def swiglu(tile_size: int = 1024, use_lut: bool = False) -> ExternalFunction:
    """SwiGLU gated activation kernel for bf16 tiles (must be 1024).

    ``out = (x * w1) * silu(x * w2)``; see [`swiglu_ref`][iron.kernels.activation.swiglu_ref].
    """
    use_lut_model = use_lut or not _arch_traits().native_tanh
    return _bf16_lut_factory(
        "swiglu",
        "swiglu_bf16",
        "swiglu.cc",
        tile_size,
        arg_arity=4,
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, In, Out),
            reference=swiglu_lut_ref if use_lut_model else swiglu_ref,
            acc_dtype=bfloat16,
            tolerance=(
                _LUT_MODEL_TOLERANCE
                if use_lut_model
                else _vtanh_family_tolerance("swiglu")
            ),
            ops_per_call=6 * tile_size,
            uses_lut=True,
        ),
        use_lut_tanh=use_lut,
    )


# aiecc measured 1600 for the polynomial branch on aie2p.
_BF16_EXP_POLY_STACK_BYTES = 2048


def bf16_exp(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise exponential kernel for bf16 tiles (must be 1024).

    Computes ``exp(clip(x, -88, 88))``: the kernel saturates rather than
    overflowing for real inputs, including infinities. On AIE2P a
    range-reduced polynomial replaces the table and rounds every result
    correctly, subnormal outputs included. See
    [`bf16_exp_ref`][iron.kernels.activation.bf16_exp_ref] for why that
    clamp matches the AIE2 table's domain.
    """
    return _bf16_lut_factory(
        "bf16_exp",
        "exp_bf16_1024",
        "bf16_exp.cc",
        tile_size,
        arg_arity=2,
        contract=_unary_lut_contract(
            bf16_exp_ref,
            count=False,
            # Only aie2's tuned branch reaches getExpBf16. The other computes a
            # range-reduced polynomial, which this model does not describe,
            # so it keeps the true-function reference and a measured bound,
            # whether or not there is a tanh instruction.
            elementwise=bf16_exp_lut_ref if _tuned_arch() == "aie2" else None,
            lut_tolerance=_EXP_LUT_TOLERANCE,
            use_lut=True,
            tolerance=_EXP_POLY_TOLERANCE,
            # The polynomial sets conv_even itself.
            setup=conv_even if _tuned_arch() == "aie2" else None,
            stack_bytes=(
                None if _tuned_arch() == "aie2" else _BF16_EXP_POLY_STACK_BYTES
            ),
        ),
    )


def exp2f_vec(tile_size: int = 1024, min_x: float = -111.0) -> ExternalFunction:
    """Software f32 ``2**x`` kernel: a minimax poly, not a LUT.

    A float32-output alternative to [`bf16_exp`]
    [iron.kernels.activation.bf16_exp] with a separately configurable
    input domain. See ``aie_kernels/activation/exp2f_vec.cc`` for the
    accuracy rationale: 9.2e-6 relative error on aie2p, 8.9e-5 on aie2.

    The same source builds for aie2.

    Args:
        tile_size: Number of elements per tile; must be a multiple of 16
            (the kernel's vector width).
        min_x: Input is clamped to this before evaluation. -126 is the
            hard floor (one f32 exponent field); on aie2p the kernel holds
            its accuracy down to it.

    Returns:
        ExternalFunction configured for the exp2f_vec kernel.

    Raises:
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
    source = _kernel_source("activation/exp2f_vec.cc")
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    return _make_extern(
        "exp2f_vec_f32",
        source,
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DEXP2F_VEC_MIN_X={float(min_x)!r}f"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            # The aie2p branch sets conv_even itself.
            setup=None if _tuned_arch() == "aie2p" else conv_even,
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=lambda x: exp2f_vec_ref(x, min_x=min_x),
            acc_dtype=np.float32,
            tolerance=Tolerance.relative(
                1e-3,
                note="measured 9.2e-6 on aie2p and 8.9e-5 on aie2; clamping "
                "[127.999, 128) costs up to 7.8e-4",
            ),
            # aiecc measured 1984 on aie2p (448 portable); remarks gives the
            # kernel 1792 on aie2 (832 portable).
            stack_bytes={"aie2": 2048, "aie2p": 2048}.get(_tuned_arch()),
        ),
    )


def tanh(tile_size: int = 1024, use_lut: bool = False) -> ExternalFunction:
    """Tanh for bf16 tiles of a positive multiple of 32 elements.

    The count is compiled in; retain
    ``tile_size`` as a trailing ``int`` argument (e.g. via
    ``transform_parallel(pass_size_to_kernel=True)``).

    Args:
        tile_size: Elements per call (a positive multiple of 32).
        use_lut: Compute tanh from the interpolated LUT rather than AIE2P's
            vtanh instruction. Moot on aie2, which only has the LUT. See
            [`tanh_lut_ref`][iron.kernels.activation.tanh_lut_ref] for what
            the LUT computes and why it is the more accurate of the two.
    """
    _require_vector_alignment("tanh", tile_size, _RUNTIME_VECTOR_WIDTH)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "tanh_bf16",
        "tanh.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DTANH_ELEMS={tile_size}"],
        contract=_unary_lut_contract(
            tanh_ref,
            count=tile_size,
            use_lut=use_lut,
            elementwise=tanh_lut_ref,
            # aie2 never reaches this: _unary_lut_contract swaps in the
            # model tolerance there, since the LUT is its only tanh.
            tolerance=_VTANH_TOLERANCE,
        ),
        use_lut_tanh=use_lut,
    )


def sigmoid(tile_size: int = 1024, use_lut: bool = False) -> ExternalFunction:
    """Sigmoid for bf16 tiles of a positive multiple of 32 elements.

    The count is compiled in; retain ``tile_size`` as a trailing ABI argument.
    """
    _require_vector_alignment("sigmoid", tile_size, _RUNTIME_VECTOR_WIDTH)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "sigmoid_bf16",
        "sigmoid.cc",
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DSIGMOID_ELEMS={tile_size}"],
        contract=_unary_lut_contract(
            sigmoid_ref,
            count=tile_size,
            use_lut=use_lut,
            elementwise=(
                sigmoid_table_ref if _tuned_arch() == "aie2p" else sigmoid_lut_ref
            ),
            tolerance=_vtanh_family_tolerance("sigmoid"),
        ),
        use_lut_tanh=use_lut,
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
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, Out, Param, Param),
            parameter_bindings=((2, tile_size),),
            reference=leaky_relu_ref,
            acc_dtype=bfloat16,
            # max(x, alpha*x) introduces exactly one rounding, on alpha*x.
            # Measured bit-exact on npu2 over 262144 elements at alpha=0.5,
            # which is a power of two and so rounds trivially; one ulp covers
            # an alpha that does not. The 0.03/0.05 with a 2% budget this
            # replaces was three orders of magnitude looser than the
            # arithmetic allows.
            tolerance=Tolerance.bf16_ulps(
                1, note="one rounding, on alpha*x; measured bit-exact at alpha=0.5"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Reference (numpy) implementations
# ---------------------------------------------------------------------------
# The kernels above are LUT approximations.  The functions below compute the
# corresponding op in numpy so host harnesses can verify the AIE output
# without each design re-implementing the math.  Output dtype matches the
# input (so a bf16 input yields a bf16 reference, comparable to the AIE
# kernel output via ``aie.utils.verify.{nearly_equal, count_mismatches}``).
# The transcendental ones work in float64 and round once to the output
# dtype: float32's exp(-x) overflows from x = -88.7 down, which made
# sigmoid and silu 0 where the true result is a nonzero bf16.


def _f64(x):
    return np.asarray(x).astype(np.float64)


def _no_neg_inf(xf):
    """``xf`` with -inf as the most negative float64.

    silu and gelu go to -0 as x goes to -inf, but at -inf itself they are
    ``-inf / inf`` and ``-inf * 0``; at any finite x the float64 arithmetic
    rounds to the limit.
    """
    return np.maximum(xf, -np.finfo(np.float64).max)


def _rounded(v, like):
    """Correctly round float64 ``v`` to ``like``'s dtype (a cast rounds twice)."""
    return round_to(v, np.asarray(like).dtype)


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
    xf = _no_neg_inf(_f64(x))
    with np.errstate(over="ignore"):
        return _rounded(xf / (1.0 + np.exp(-xf)), x)


def gelu_ref(x):
    """Numpy reference for [`gelu`][iron.kernels.activation.gelu].

    Tanh approximation ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
    Matches the C++ kernel's tanh-GELU formula. It is evaluated in float64:
    in float32, ``1 + tanh`` cancels for x below about -4.5 and leaves
    values over ten times too large.
    """
    xf = _no_neg_inf(x.astype(np.float64))
    with np.errstate(over="ignore"):
        inner = math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf**3)
    return (0.5 * xf * (1.0 + np.tanh(inner))).astype(x.dtype)


def bf16_exp_lut_ref(x):
    """Model of ``getExpBf16``, the LUT exponential the bf16_exp kernel uses.

    The kernel clamps to ``+/-_EXP_BF16_CLAMP``, converts to Q8 with a floor
    (``bfloat16_to_int(x, 8)``), then reads the byte halves of that fixed-point
    key as two table indices and multiplies: ``exp(x) = exp(int) * exp(frac)``.
    Both tables hold exactly ``bfloat16(exp(.))``, so they are written here as
    that rule rather than as 512 opaque floats -- the unreachable middle of the
    integer table (keys the clamp cannot produce) is the only part that is not
    an exponential, and it is never read.

    The product is exact in f32 (two bf16 operands), so the model differs from
    the device only where the hardware flushes the one subnormal table entry,
    ``bfloat16(exp(-88))``.
    """
    xf = np.clip(np.asarray(x).astype(np.float32), -_EXP_BF16_CLAMP, _EXP_BF16_CLAMP)
    key = np.floor(xf * 256.0).astype(np.int32).astype(np.int16).astype(np.uint16)
    i = np.arange(256)
    ilut = np.where(
        i <= 88,
        np.exp(np.minimum(i, 88.0)),
        np.where(i >= 168, np.exp(i - 256.0), np.exp(88.0)),
    )
    ilut = np.asarray(ilut, np.float32).astype(bfloat16).astype(np.float32)
    flut = np.exp(np.arange(256) / 256.0).astype(np.float32).astype(bfloat16)
    return (ilut[key >> 8] * flut.astype(np.float32)[key & 255]).astype(np.float32)


def _bf16(v):
    """Round an f32 array to bf16 and back, as an accumulator store does."""
    return np.asarray(v, np.float32).astype(bfloat16).astype(np.float32)


def _bf16_ftz(v):
    """Round an f32 product to bf16 as an accumulator store does.

    A subnormal product is flushed to zero first, keeping its sign, so it
    does not round to a subnormal bf16 or up to 2**-126.
    """
    v = np.asarray(v, np.float32)
    return _bf16(np.where(np.abs(v) < 2.0**-126, np.copysign(np.float32(0), v), v))


def sigmoid_lut_ref(x):
    """Model of [`sigmoid`][iron.kernels.activation.sigmoid]'s LUT build on aie2.

    AIE2P's reads a table of its own; see
    [`sigmoid_table_ref`][iron.kernels.activation.sigmoid_table_ref]. This
    follows activation/sigmoid.cc's aie2 branch step for step: ``x/2`` is
    exact (0.5 is a power of two), the accumulator overload of ``tanh_bf16_v16`` narrows to bf16
    before the table, and the ``+1`` and ``*0.5`` stay in the accumulator so
    there is a single store rounding at the end.
    """
    xf = np.asarray(x).astype(np.float32)
    t = np.asarray(tanh_lut_ref(_bf16(xf * 0.5)), np.float32)
    return _bf16((t + 1.0) * 0.5).astype(np.asarray(x).dtype)


def sigmoid_table_ref(x):
    """Model of AIE2P's [`sigmoid`][iron.kernels.activation.sigmoid] built with ``use_lut=True``.

    activation/sigmoid.cc reads its own table there: getTanhBf16's segments
    rewritten for ``0.5 + 0.5 * tanh(x/2)``, so segment ``e`` has slope
    ``slope[e] / 4`` and offset ``0.5 + 0.5 * offset[e]`` and covers 0.5 of x.
    x is clamped to ``[-8, 8 - 1/32]``; the one rounding is the store to bf16,
    as in [`tanh_lut_ref`][iron.kernels.activation.tanh_lut_ref].
    [`sigmoid_lut_ref`][iron.kernels.activation.sigmoid_lut_ref] also rounds
    ``tanh`` to bf16 before ``0.5 * (1 + t)``, which makes it 0 over
    ``[-7.5, -6.9]``, where this is not.
    """
    xf = np.clip(np.asarray(x).astype(np.float32), -8.0, 8.0 - 1.0 / 32)
    e = np.clip(np.floor(xf * 2.0).astype(np.int64), -16, 15) + 16
    slope = np.asarray(_TANH_LUT_SLOPE, np.float32)[e] / 4
    offset = 0.5 + 0.5 * np.asarray(_TANH_LUT_OFFSET, np.float32)[e]
    return (slope * xf + offset).astype(bfloat16).astype(np.asarray(x).dtype)


def silu_lut_ref(x):
    """Model of [`silu`][iron.kernels.activation.silu]'s LUT build on aie2.

    AIE2P's multiplies by its sigmoid table instead; see
    [`silu_table_ref`][iron.kernels.activation.silu_table_ref].
    activation/silu.cc narrows the sigmoid factor to bf16 before the final
    multiply, so that rounding is modelled too, not folded away. The sigmoid
    is exactly 0 from x = -8 down, and x is clamped there before the multiply,
    so -inf gives 0 rather than NaN. A subnormal product is flushed to zero.
    """
    xf = np.asarray(x).astype(np.float32)
    sig = np.asarray(sigmoid_lut_ref(xf), np.float32)
    return _bf16_ftz(np.maximum(xf, -8.0) * sig).astype(np.asarray(x).dtype)


def silu_table_ref(x):
    """Model of AIE2P's [`silu`][iron.kernels.activation.silu] built with ``use_lut=True``.

    [`silu_lut_ref`][iron.kernels.activation.silu_lut_ref] with AIE2P's
    sigmoid table,
    [`sigmoid_table_ref`][iron.kernels.activation.sigmoid_table_ref], as the
    bf16 factor.
    """
    xf = np.asarray(x).astype(np.float32)
    sig = np.asarray(sigmoid_table_ref(xf), np.float32)
    return _bf16_ftz(np.maximum(xf, -8.0) * sig).astype(np.asarray(x).dtype)


def swiglu_lut_ref(x, w1, w2):
    """Model of [`swiglu`][iron.kernels.activation.swiglu] built with ``use_lut=True``.

    activation/swiglu.cc narrows after every multiply -- ``x*w1``, ``x*w2``, the
    sigmoid factor and the silu product each land in a bf16 register before
    the next step -- which is what this reproduces. ``x*w2`` is clamped at -8
    before its multiply, as in silu_lut_ref, and where the silu product is 0
    the output is 0, so an overflowed ``x*w1`` does not make ``inf * 0``.
    Each subnormal product is flushed to zero, so a subnormal ``x*w2`` zeroes
    the output through the gate.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        xw1 = _bf16_ftz(np.asarray(x, np.float32) * np.asarray(w1, np.float32))
        xw2 = _bf16_ftz(np.asarray(x, np.float32) * np.asarray(w2, np.float32))
        sig = np.asarray(sigmoid_lut_ref(xw2), np.float32)
        silu_out = _bf16_ftz(np.maximum(xw2, -8.0) * sig)
        out = np.where(silu_out == 0, np.float32(0.0), _bf16_ftz(xw1 * silu_out))
    return out.astype(np.asarray(x).dtype)


def tanh_lut_ref(x):
    """Numpy model of ``getTanhBf16``, the interpolated-LUT tanh.

    The kernel clamps x to the table's range ``[-4, 4 - 1/64]``, then evaluates
    ``slope[e] * x + offset[e]`` for ``e = floor(4x) + 16``: 32 segments of
    width 0.25 over ``[-4, 4)``. The end segments are the constants -1 and +1,
    so the clamp changes no finite result, and +-inf gives +-1 rather than
    ``0 * inf``. The product is exact in f32 (bf16 carries 8 mantissa bits and
    8 + 8 < 24), so the only rounding is the accumulator's store back to bf16,
    which is why the build using this is judged at one ulp rather than a
    percentage.

    This is what [`tanh`][iron.kernels.activation.tanh] computes with
    ``use_lut=True``, and what it always computes on aie2. The default aie2p
    build uses the ``vtanh`` instruction instead, which is a coarser
    approximation with no published spec, so it is judged against
    [`tanh_ref`][iron.kernels.activation.tanh_ref] and a measured bound.
    """
    xf = np.clip(np.asarray(x).astype(np.float32), -4.0, 4.0 - 1.0 / 64)
    e = np.clip(np.floor(xf * 4.0).astype(np.int64), -16, 15) + 16
    slope = np.asarray(_TANH_LUT_SLOPE, np.float32)[e]
    offset = np.asarray(_TANH_LUT_OFFSET, np.float32)[e]
    return (slope * xf + offset).astype(bfloat16).astype(np.asarray(x).dtype)


def tanh_ref(x):
    """Numpy reference for [`tanh`][iron.kernels.activation.tanh] — element-wise ``tanh(x)``.

    LUT/native-approximation territory; pair with ``rtol=0.128`` when verifying.
    """
    return _rounded(np.tanh(_f64(x)), x)


def sigmoid_ref(x):
    """Numpy reference for [`sigmoid`][iron.kernels.activation.sigmoid] — ``1 / (1 + exp(-x))``.

    LUT-approximation territory; pair with ``rtol=0.128`` when verifying.
    """
    with np.errstate(over="ignore"):
        return _rounded(1.0 / (1.0 + np.exp(-_f64(x))), x)


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
    float64; LUT-approximation territory, pair with ``rtol=0.128``.

    Where silu is 0 (``x * w2`` is -inf, or low enough to underflow) the
    output is 0, as swiglu_lut_ref's is, even where ``x * w1`` overflowed:
    0 is the limit as x goes to -inf, and ``inf * 0`` would be NaN.
    """
    xf = _f64(x)
    xw1 = _f64(round_to(xf * _f64(w1), bfloat16))
    xw2 = _no_neg_inf(_f64(round_to(xf * _f64(w2), bfloat16)))
    with np.errstate(over="ignore", invalid="ignore"):
        silu = xw2 / (1.0 + np.exp(-xw2))
        out = np.where(silu == 0, np.copysign(0.0, xw1) * silu, xw1 * silu)
    return _rounded(out, x)


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
    The clamp also keeps the result in range: ``exp(88) = 1.65e+38`` fits
    bf16 and float32 where ``exp(89)`` would not.
    """
    xf = np.clip(_f64(x), -_EXP_BF16_CLAMP, _EXP_BF16_CLAMP)
    # The clamp rules out overflow, so that warning stays un-suppressed and
    # would now be a real signal. A NaN input still reaches exp -- the kernel
    # declares nonfinite="unspecified" and callers do feed raw bit patterns
    # (programming_examples/basic/vector_exp sweeps all 65536 of them).
    with np.errstate(invalid="ignore"):
        return _rounded(np.exp(xf), x)


def exp2f_vec_ref(x, min_x: float = -111.0):
    """Numpy reference for [`exp2f_vec`][iron.kernels.activation.exp2f_vec]: exact ``2**x``.

    Unlike the LUT-based refs above, this is float64 ``2**x`` (not a
    reimplementation of the on-device poly): the kernel holds 8.9e-5
    relative error or better, several orders tighter than the LUT-based
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
    way before applying the softmax, in float64.  ``x.size`` must be a
    multiple of ``tile_size``.
    """
    xf = _f64(x)
    if xf.size % tile_size != 0:
        raise ValueError(
            f"softmax_ref: x has {xf.size} elements; not a multiple of "
            f"tile_size={tile_size}"
        )
    flat = xf.reshape(-1, tile_size)
    flat = flat - flat.max(axis=1, keepdims=True)
    exp = np.exp(flat)
    out = exp / exp.sum(axis=1, keepdims=True)
    return _rounded(out.reshape(x.shape), x)
