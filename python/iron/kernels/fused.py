# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Single-tile composition of the fused GEMM init, reduction and drain ABI."""

import hashlib
from pathlib import Path

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils import bfp
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    ARCH_TRAITS,
    KernelContract,
    Param,
    TensorLayout,
    Trace,
    _detect_arch,
    _include_dirs,
    _kernel_source,
    _portable_flags,
)
from .activation import _bf16_ulp, _vtanh_error
from .quant import _bf16_floor

# Largest |d/dx| of each epilogue: x * sigmoid(kx) peaks at 1.0998.
_SLOPE = {"none": 1.0, "sigmoid": 0.25, "silu": 1.1, "gelu": 1.1}
_MODES = {"none": 0, "gelu": 1, "silu": 2, "sigmoid": 3}
# (bfloat16)1.702f, the scale gelu_bf16_steps_vec multiplies by.
_GELU_SCALE_BF16 = 1.703125


class _FusedMMKernel(ExternalFunction):
    """``fused_mm_tile`` with the entry points it calls, bound by C symbol."""

    fused_mm_tile: ExternalFunction
    mm_fused_acc_init: Kernel
    mm_fused_k_step: Kernel
    mm_fused_epilogue_chunk: Kernel


def _bf16_round(x, rounding):
    """Narrow to bf16 as a core does in ``rounding``, returned as float64."""
    x = np.asarray(x, np.float32)
    narrowed = _bf16_floor(x) if rounding == "floor" else x.astype(bfloat16)
    return narrowed.astype(np.float64)


def fused_mm(
    *,
    dim_m=32,
    dim_k=32,
    dim_n=16,
    band_m=16,
    chunk_k=16,
    out_chunk=64,
    epilogue="none",
    clamp=None,
    bfp16_b=False,
    b_col_maj=False,
    emulate_bf16_mmul_with_bfp16=False,
    epilogue_modes=None,
    rounding="conv_even",
    gelu="fp32",
    mmul_shape=None,
    c_depth=2,
    step_markers=False,
) -> _FusedMMKernel:
    """Compute one bf16 ``A @ B`` tile, with an f32 reduction and fused epilogue.

    This bounded composition holds its operands and accumulator on one core;
    it is not the streaming whole-matrix operator. A and B use bf16 storage
    on both architectures unless ``bfp16_b`` is set.
    ``band_m`` and ``chunk_k`` subdivide the reduction; ``out_chunk`` subdivides
    the drain with a depth of two. The contract supplies the blocked storage
    layouts, including B's column-major ordering of row-major microblocks.

    ``epilogue`` is ``none``, ``gelu`` (the kernel's sigmoid approximation,
    not the tanh GELU curve), ``silu`` or ``sigmoid``. An optional finite
    ``(min, max)`` clamp follows the activation, before the bf16 conversion.

    ``bfp16_b`` (aie2p only) selects the prepacked-B form amd/IRON's flm GEMM
    builds: B arrives as bfp16ebs8 blocks the host packs once, the mmul is
    8x8x8, and the core converts A to bfp16 itself. The host rounds B in the
    ``rounding`` mode, as IRON's ``pack_b`` does, and the reference
    multiplies both operands as the core sees them.

    ``b_col_maj`` (with an ``(r, 8, 8)`` mmul, r 8 on aie2p) takes B as stored
    transposed, the ``(n, k)`` layout of a checkpoint's weights: each k chunk
    holds k panels of ``s`` rows, every n row of a panel ``s`` contiguous k,
    a layout a DMA builds from ``(n, k)`` storage in 16-byte runs. On bf16
    macs the core transposes each half block in one shuffle; on bfp16 macs
    it converts each block as loaded.

    ``emulate_bf16_mmul_with_bfp16`` (aie2p) multiplies on bfp16 macs, as
    aie_api's emulated bf16 mmul does: the core converts A and B to
    bfp16ebs8, each shared exponent spanning 8 consecutive k, and the
    reference multiplies both operands as the core sees them. The mmul
    defaults to 8x8x8, which ``b_col_maj`` needs. ``bfp16_b`` implies it.

    ``epilogue_modes`` lists the activations compiled in, for a caller that
    selects one at run time through ``mm_fused_epilogue_chunk``'s mode
    argument. It defaults to ``(epilogue,)``. ``epilogue`` is the mode
    ``fused_mm_tile`` and the reference use, and must be in the list. On
    aie2 an activation links the tanh tables, 5 KB of L1; ``("none",)``
    links none.

    ``rounding`` is the core's rounding mode for every f32 -> bf16 and
    bf16 -> bfp16 conversion: ``conv_even`` or ``floor``, the mode a core
    powers up in.

    ``gelu`` selects how the gelu mode computes. ``fp32`` applies it to the
    f32 accumulator and rounds once. ``bf16_steps`` rounds the accumulator to
    bf16 and rounds again after each step of ``x * sigmoid(1.702x)``, as
    FastFlowLM's shipped mm overlay does. With ``rounding="floor"`` it
    reproduces that overlay bit for bit on AIE2P.

    ``mmul_shape`` overrides the ``(r, s, t)`` of the mmul. ``c_depth`` is
    the depth of the C fifo, which a caller's drain loop unrolls by.
    ``step_markers`` brackets each init, k step and drain chunk with event
    markers, for a caller that calls those entry points itself. Otherwise one
    pair brackets the whole ``fused_mm_tile`` call.

    The returned function carries every entry point of its object as an
    attribute named by the C symbol: ``fn.mm_fused_acc_init``,
    ``fn.mm_fused_k_step``, ``fn.mm_fused_epilogue_chunk`` and
    ``fn.fused_mm_tile`` (``fn`` itself).
    """
    arch = _detect_arch()
    if bfp16_b and not ARCH_TRAITS[arch].bfp16:
        raise ValueError("fused_mm: bfp16_b needs aie2p; bfp16ebs8 is an AIE2P type")
    if emulate_bf16_mmul_with_bfp16 and not ARCH_TRAITS[arch].bfp16:
        raise ValueError("fused_mm: emulate_bf16_mmul_with_bfp16 needs aie2p")
    if b_col_maj and bfp16_b:
        raise ValueError("fused_mm: b_col_maj reads a bf16 B; bfp16_b's is packed")
    emulated = bfp16_b or emulate_bf16_mmul_with_bfp16
    if emulated:
        r, s, t = 8, 8, 8
    else:
        r, s, t = (4, 8, 4) if arch == "aie2" else (4, 8, 8)
    if mmul_shape is not None:
        if bfp16_b and tuple(mmul_shape) != (8, 8, 8):
            raise ValueError("fused_mm: bfp16_b needs mmul_shape (8, 8, 8)")
        r, s, t = mmul_shape
    if b_col_maj and (s, t) != (8, 8):
        raise ValueError("fused_mm: b_col_maj needs an (r, 8, 8) mmul_shape")
    if b_col_maj and arch == "aie2p" and r != 8:
        raise ValueError("fused_mm: b_col_maj on aie2p needs mmul (8, 8, 8)")
    dims = (dim_m, dim_k, dim_n, band_m, chunk_k, out_chunk, c_depth, r, s, t)
    if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims):
        raise ValueError("fused_mm dimensions must be positive integers")
    if (
        dim_m % band_m
        or band_m % (2 * r)
        or dim_n % (2 * t)
        or dim_k % chunk_k
        or chunk_k % s
        or out_chunk % 16
        or (dim_m * dim_n) % (c_depth * out_chunk)
    ):
        raise ValueError("fused_mm dimensions violate band, mmul or drain divisibility")
    modes = _MODES
    if epilogue not in modes:
        raise ValueError(f"unknown fused_mm epilogue: {epilogue}")
    if epilogue_modes is None:
        epilogue_modes = (epilogue,)
    if any(m not in modes for m in epilogue_modes):
        raise ValueError(f"unknown fused_mm epilogue in {epilogue_modes}")
    if epilogue not in epilogue_modes:
        raise ValueError(f"epilogue {epilogue} is not in epilogue_modes")
    if rounding not in ("conv_even", "floor"):
        raise ValueError(f"fused_mm: rounding must be conv_even or floor: {rounding}")
    if gelu not in ("fp32", "bf16_steps"):
        raise ValueError(f"fused_mm: gelu must be fp32 or bf16_steps: {gelu}")
    steps = gelu == "bf16_steps" and epilogue == "gelu"
    if clamp is not None:
        if len(clamp) != 2 or not np.isfinite(clamp).all() or clamp[0] > clamp[1]:
            raise ValueError("clamp must be a finite (min, max) pair with min <= max")
        clamp = tuple(float(v) for v in clamp)

    def pack_a(a):
        return (
            a.reshape(-1, dim_m // r, r, dim_k // chunk_k, chunk_k // s, s)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(len(a), -1)
        )

    def unpack_a(a):
        return (
            a.reshape(-1, dim_k // chunk_k, dim_m // r, chunk_k // s, r, s)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(-1, dim_m, dim_k)
        )

    def pack_b(b):
        return (
            b.reshape(-1, dim_k // chunk_k, chunk_k // s, s, dim_n // t, t)
            .transpose(0, 1, 4, 2, 3, 5)
            .reshape(len(b), -1)
        )

    def unpack_b(b):
        return (
            b.reshape(-1, dim_k // chunk_k, dim_n // t, chunk_k // s, s, t)
            .transpose(0, 1, 3, 4, 2, 5)
            .reshape(-1, dim_k, dim_n)
        )

    # Block (i, j) of a k chunk at (i * colB + j), each B^T: n-major.
    def pack_b_t(b):
        return (
            b.reshape(-1, dim_k // chunk_k, chunk_k // s, s, dim_n // t, t)
            .transpose(0, 1, 2, 4, 5, 3)
            .reshape(len(b), -1)
        )

    def unpack_b_t(b):
        return (
            b.reshape(-1, dim_k // chunk_k, chunk_k // s, dim_n // t, t, s)
            .transpose(0, 1, 2, 5, 3, 4)
            .reshape(-1, dim_k, dim_n)
        )

    # The bfp16 blocks are t-major (block (i, j) holds B^T), because
    # mac_8x8_8x8T takes B transposed; that also makes each shared exponent
    # span 8 consecutive k of one column, the grouping the mac expects. The
    # blocks round as amd/IRON's weight packer does.
    def pack_b_bfp(b):
        blocks = (
            b.reshape(-1, dim_k // chunk_k, chunk_k // s, s, dim_n // t, t)
            .transpose(0, 1, 4, 2, 5, 3)
            .reshape(len(b), -1)
        )
        return bfp.encode(blocks, rounding=rounding)

    def unpack_b_bfp(b):
        return (
            bfp.decode(b)
            .reshape(-1, dim_k // chunk_k, dim_n // t, chunk_k // s, t, s)
            .transpose(0, 1, 3, 5, 2, 4)
            .reshape(-1, dim_k, dim_n)
        )

    def pack_c(c):
        return (
            c.reshape(-1, dim_m // r, r, dim_n // t, t)
            .transpose(0, 1, 3, 2, 4)
            .reshape(len(c), -1)
        )

    def unpack_c(c):
        return (
            c.reshape(-1, dim_m // r, dim_n // t, r, t)
            .transpose(0, 1, 3, 2, 4)
            .reshape(-1, dim_m, dim_n)
        )

    def operands(a, b):
        a = a.reshape(-1, dim_m, dim_k).astype(np.float32)
        b = b.reshape(-1, dim_k, dim_n).astype(np.float32)
        if emulated:
            a = bfp.quantize(a, rounding=rounding)
        if bfp16_b:
            b = unpack_b_bfp(pack_b_bfp(b))
        elif emulated:
            b = bfp.quantize(b.swapaxes(1, 2), rounding=rounding).swapaxes(1, 2)
        return a.astype(np.float64), b.astype(np.float64)

    # float64 throughout: in float32, 1 + tanh cancels for large negative
    # inputs and the product's rounding depends on numpy's summation order.
    # gelu_bf16_steps_vec with an exact tanh.
    def gelu_steps(c):
        x = _bf16_round(c, rounding)
        y = _bf16_round(x * _GELU_SCALE_BF16, rounding)
        sig = _bf16_round(np.tanh(y * 0.5) + 1, rounding) * 0.5
        return _bf16_round(x * sig, rounding)

    def activate(c):
        if steps:
            c = gelu_steps(c)
        elif epilogue != "none":
            x = c * 1.702 if epilogue == "gelu" else c
            sigmoid = (np.tanh(x * 0.5) + 1) * 0.5
            c = sigmoid if epilogue == "sigmoid" else c * sigmoid
        if clamp is not None:
            c = np.clip(c, clamp[0], clamp[1])
        return c

    def reference(a, b):
        a, b = operands(a, b)
        c = activate(a @ b)
        return c.reshape(len(c), -1)

    # Each step of gelu_steps rounds in the core and in the reference. A
    # rounding moves its result by under one ulp of the larger operand, so
    # the step's input error plus that ulp bounds its output error. tanh is
    # 1-Lipschitz and vtanh adds its own error.
    def gelu_steps_bound(c, acc_err):
        def ulp(v, dv):
            return _bf16_ulp(np.abs(v) + dv)

        x = _bf16_round(c, rounding)
        dx = acc_err + ulp(x, acc_err)
        y = _bf16_round(x * _GELU_SCALE_BF16, rounding)
        dy = _GELU_SCALE_BF16 * dx + ulp(y, _GELU_SCALE_BF16 * dx)
        # vtanh's error is not monotone, so take it at both ends of the
        # argument's interval and at its centre.
        vtanh = np.maximum.reduce(
            [_vtanh_error(0.5 * (np.abs(y) + d)) for d in (-dy, 0, dy)]
        )
        dt = 0.5 * dy + vtanh
        sig = _bf16_round(np.tanh(y * 0.5) + 1, rounding)
        dsig = 0.5 * (dt + ulp(sig, dt))
        out = x * sig * 0.5
        dout = 0.5 * sig * dx + np.abs(x) * dsig + dx * dsig
        return dout + ulp(out, dout) + 2.0**-126

    # Floor rounding errs by up to one f32 ulp per accumulation, not half.
    acc_ulps = 2.0**-23 if rounding == "floor" else 2.0**-24

    # The core's f32 sums are off by at most dim_k f32 ulps of sum |a*b|,
    # and the activation's slope (under 1.1) carries that through. On
    # AIE2P tanh is vtanh (see activation._vtanh_error): sigmoid scales
    # its error by 1/2 and x * sigmoid(u) by |x|/2. Every other step is an
    # exact bf16 product or an f32 add, and one output ulp covers the store.
    def error_bound(a, b):
        a, b = operands(a, b)
        c = a @ b
        if steps:
            acc_err = dim_k * acc_ulps * (np.abs(a) @ np.abs(b))
            return gelu_steps_bound(c, acc_err).reshape(len(c), -1)
        err = dim_k * acc_ulps * (np.abs(a) @ np.abs(b)) * _SLOPE[epilogue]
        if epilogue == "sigmoid":
            err = err + 0.5 * _vtanh_error(0.5 * c)
        elif epilogue != "none":
            u = 0.851 * c if epilogue == "gelu" else 0.5 * c
            err = err + 0.5 * np.abs(c) * _vtanh_error(u)
        err = err + _bf16_ulp(activate(c)) + 2.0**-126
        return err.reshape(len(c), -1)

    flags = {
        "TILE_M": dim_m,
        "TILE_MA": band_m,
        "TILE_K": dim_k,
        "TILE_N": dim_n,
        "CT_K": chunk_k,
        "R": r,
        "S": s,
        "T": t,
        "OUT_CHUNK": out_chunk,
        "C_DEPTH": c_depth,
        # The kernel selects the activation at runtime; the mask only decides
        # which bodies are compiled in, so each mode costs program memory.
        "EPILOGUE_MODE_MASK": sum(1 << modes[m] for m in set(epilogue_modes)),
    }
    compile_flags = [
        *(["-DROUND_CONV_EVEN"] if rounding == "conv_even" else []),
        *(f"-DMM_FUSED_{name}={value}" for name, value in flags.items()),
        *_portable_flags(),
    ]
    if emulated:
        compile_flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    if bfp16_b:
        compile_flags.append("-DMM_FUSED_BFP16_B")
    if b_col_maj:
        compile_flags.append("-DMM_FUSED_B_COL_MAJ")
    if gelu == "bf16_steps":
        compile_flags.append("-DMM_FUSED_GELU_BF16_STEPS")
    if step_markers:
        compile_flags.append("-DMM_FUSED_STEP_MARKERS")
    # An absent clamp is (-inf, +inf), which leaves every finite value
    # untouched, so there is no unclamped path to select between.
    bounds = clamp if clamp is not None else (-np.inf, np.inf)
    clamp_bits = tuple(int(np.float32(v).view(np.int32)) for v in bounds)
    source = _kernel_source("fused/fused_mm_tile.cc")
    include_dirs = _include_dirs()
    native_tanh = ARCH_TRAITS[arch].native_tanh
    if not native_tanh:
        from aie.utils import config

        runtime = Path(config.aie_runtime_lib_dir()) / arch.upper()
        include_dirs.append(str(runtime))
    # Include the complete recipe, not just geometry: architecture, source
    # location and runtime includes can change without changing the operands.
    key = (
        "fused_mm_tile",
        source,
        tuple(include_dirs),
        tuple(compile_flags),
        False,
        arch,
        # The clamp is no longer a compile flag, so two clamps of the same
        # kernel share compile_flags. They still need their own bindings,
        # reference and tolerance, so the bounds belong in the key.
        clamp_bits,
    )
    # With one mode compiled in, the mask names the bound mode. With several,
    # the bound mode needs its own place in the key, for the clamp's reason.
    if len(set(epilogue_modes)) > 1:
        key += (modes[epilogue],)
    prefix = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    if bfp16_b:
        b_codec = (pack_b_bfp, unpack_b_bfp)
    elif b_col_maj:
        b_codec = (pack_b_t, unpack_b_t)
    else:
        b_codec = (pack_b, unpack_b)
    contract = KernelContract(
        trace=(
            Trace.partial("markers bracket each init, k step and drain chunk")
            if step_markers
            else Trace.whole_call()
        ),
        roles=(In, In, Out, Param, Param, Param),
        # The operands are held on the core in this blocking; nothing is
        # streamed transformed, the host packs them (block, no stream).
        layouts=(
            TensorLayout((dim_m, dim_k), pack_a, unpack_a, block=(r, s)),
            TensorLayout((dim_k, dim_n), *b_codec, block=(s, t)),
            TensorLayout((dim_m, dim_n), pack_c, unpack_c, block=(r, t)),
            None,
            None,
            None,
        ),
        # Bound here rather than left to the caller: `epilogue` and `clamp`
        # stay factory arguments, so `reference` below closes over the same
        # values the core is given.
        parameter_bindings=(
            (3, modes[epilogue]),
            (4, clamp_bits[0]),
            (5, clamp_bits[1]),
        ),
        reference=reference,
        # Without a tanh instruction the epilogue reads getTanhBf16's table,
        # which this box cannot measure.
        tolerance=(
            Tolerance.relative(
                0.02 if epilogue == "none" else 0.04,
                0.01 if epilogue == "none" else 0.04,
                note="bf16 store; activated path additionally narrows tanh to bf16",
            )
            if not native_tanh
            else Tolerance.bounded(
                error_bound,
                note=(
                    "f32 accumulation, vtanh's error measured on npu2 and one "
                    "bf16 ulp per rounding step, carried through each step"
                    if steps
                    else "f32 accumulation, vtanh's error measured on npu2 and "
                    "one bf16 store ulp, per output; fails a 1.5% change to "
                    "gelu's 1.702 and a 3% change to silu's or sigmoid's 1/2"
                ),
            )
        ),
        acc_dtype=np.float32,
        reduction=dim_k,
        ops_per_call=2 * dim_m * dim_k * dim_n,
        # An aie2 activation reaches getTanhBf16; aie2p has no table, and
        # without an activation the source links none.
        uses_lut=any(m != "none" for m in epilogue_modes),
    )
    fn = _FusedMMKernel(
        "fused_mm_tile",
        source_file=str(source),
        arg_types=[
            np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]],
            (
                np.ndarray[(dim_k * dim_n // 8,), np.dtype[v8bfp16ebs8]]
                if bfp16_b
                else np.ndarray[(dim_k * dim_n,), np.dtype[bfloat16]]
            ),
            np.ndarray[(dim_m * dim_n,), np.dtype[bfloat16]],
            np.int32,
            np.int32,
            np.int32,
        ],
        include_dirs=include_dirs,
        compile_flags=compile_flags,
        # Object compilation renames every defined symbol, including the
        # included init/k_step/epilogue, zero kernels and AIE2 LUT exports.
        symbol_prefix=prefix,
        contract=contract,
    )
    acc = np.ndarray[(dim_m * dim_n,), np.dtype[np.float32]]
    b_chunk = (
        np.ndarray[(chunk_k * dim_n // 8,), np.dtype[v8bfp16ebs8]]
        if bfp16_b
        else np.ndarray[(chunk_k * dim_n,), np.dtype[bfloat16]]
    )
    bind = fn.object_file.bind
    fn.fused_mm_tile = fn
    fn.mm_fused_acc_init = bind("mm_fused_acc_init", [acc])
    # The trailing int32 is the A band.
    fn.mm_fused_k_step = bind(
        "mm_fused_k_step",
        [np.ndarray[(band_m * chunk_k,), np.dtype[bfloat16]], b_chunk, acc, np.int32],
    )
    # outer, half, mode, clamp_min_bits, clamp_max_bits
    fn.mm_fused_epilogue_chunk = bind(
        "mm_fused_epilogue_chunk",
        [np.ndarray[(out_chunk,), np.dtype[bfloat16]], acc] + [np.int32] * 5,
    )
    return fn
