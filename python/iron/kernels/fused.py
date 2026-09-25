# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Single-tile composition of the fused GEMM init, reduction and drain ABI."""

import hashlib
from pathlib import Path

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron.kernel import ExternalFunction
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
    _device,
    _include_dirs,
    _kernel_source,
    _portable_flags,
)
from .activation import _bf16_ulp, _vtanh_error

# Largest |d/dx| of each epilogue: x * sigmoid(kx) peaks at 1.0998.
_SLOPE = {"none": 1.0, "sigmoid": 0.25, "silu": 1.1, "gelu": 1.1}


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
) -> ExternalFunction:
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
    8x8x8, and the core converts A to bfp16 itself. The host rounds B to
    nearest-even, as IRON's ``pack_b`` does, and the reference multiplies
    both operands as the core sees them.
    """
    arch = _detect_arch()
    device = _device()
    if bfp16_b and not ARCH_TRAITS[arch].bfp16:
        raise ValueError("fused_mm: bfp16_b needs aie2p; bfp16ebs8 is an AIE2P type")
    if bfp16_b:
        r, s, t = 8, 8, 8
    else:
        r, s, t = (4, 8, 4) if arch == "aie2" else (4, 8, 8)
    dims = (dim_m, dim_k, dim_n, band_m, chunk_k, out_chunk)
    if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in dims):
        raise ValueError("fused_mm dimensions must be positive integers")
    if (
        dim_m % band_m
        or band_m % (2 * r)
        or dim_n % (2 * t)
        or dim_k % chunk_k
        or chunk_k % s
        or out_chunk % 16
        or (dim_m * dim_n) % (2 * out_chunk)
    ):
        raise ValueError("fused_mm dimensions violate band, mmul or drain divisibility")
    modes = {"none": 0, "gelu": 1, "silu": 2, "sigmoid": 3}
    if epilogue not in modes:
        raise ValueError(f"unknown fused_mm epilogue: {epilogue}")
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
        return bfp.encode(blocks, rounding="conv_even")

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
        if bfp16_b:
            # The core converts A itself; B is whatever the host packed.
            a = bfp.quantize(a, rounding="conv_even")
            b = unpack_b_bfp(pack_b_bfp(b))
        return a.astype(np.float64), b.astype(np.float64)

    # float64 throughout: in float32, 1 + tanh cancels for large negative
    # inputs and the product's rounding depends on numpy's summation order.
    def activate(c):
        if epilogue != "none":
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

    # The core's f32 sums are off by at most dim_k f32 ulps of sum |a*b|,
    # and the activation's slope (under 1.1) carries that through. On
    # AIE2P tanh is vtanh (see activation._vtanh_error): sigmoid scales
    # its error by 1/2 and x * sigmoid(u) by |x|/2. Every other step is an
    # exact bf16 product or an f32 add, and one output ulp covers the store.
    def error_bound(a, b):
        a, b = operands(a, b)
        c = a @ b
        err = dim_k * 2.0**-24 * (np.abs(a) @ np.abs(b)) * _SLOPE[epilogue]
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
        "C_DEPTH": 2,
        # The kernel selects the activation at runtime; the mask only decides
        # which bodies are compiled in. Admitting just this one keeps the
        # program memory of a single-activation design at its old size.
        "EPILOGUE_MODE_MASK": 1 << modes[epilogue],
    }
    compile_flags = [
        "-DROUND_CONV_EVEN",
        *(f"-DMM_FUSED_{name}={value}" for name, value in flags.items()),
        *_portable_flags(),
    ]
    if bfp16_b:
        # amd/IRON's pair: the first selects aie_api's bfp16-emulated bf16
        # mmul, the second the prepacked B storage.
        compile_flags += [
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DMM_FUSED_BFP16_B",
        ]
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
        device.default_core_stack_bytes,
        # The clamp is no longer a compile flag, so two clamps of the same
        # kernel share compile_flags. They still need their own bindings,
        # reference and tolerance, so the bounds belong in the key.
        clamp_bits,
    )
    prefix = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    contract = KernelContract(
        trace=Trace.whole_call(),
        roles=(In, In, Out, Param, Param, Param),
        # The operands are held on the core in this blocking; nothing is
        # streamed transformed, the host packs them (block, no stream).
        layouts=(
            TensorLayout((dim_m, dim_k), pack_a, unpack_a, block=(r, s)),
            (
                TensorLayout((dim_k, dim_n), pack_b_bfp, unpack_b_bfp, block=(s, t))
                if bfp16_b
                else TensorLayout((dim_k, dim_n), pack_b, unpack_b, block=(s, t))
            ),
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
                note="f32 accumulation, vtanh's error measured on npu2 and "
                "one bf16 store ulp, per output; fails a 1.5% change to "
                "gelu's 1.702 and a 3% change to silu's or sigmoid's 1/2",
            )
        ),
        acc_dtype=np.float32,
        reduction=dim_k,
        ops_per_call=2 * dim_m * dim_k * dim_n,
        # The aie2 epilogue reaches getTanhBf16; aie2p has no table.
        uses_lut=True,
        # Reserve the f32 accumulator plus call frames and epilogue spills:
        # AIE2P SiLU with clamp needs 1600 bytes beyond the accumulator.
        # aiecc still checks the measured linked stack.
        stack_bytes=np.dtype(np.float32).itemsize * dim_m * dim_n
        + max(device.default_core_stack_bytes, 2048),
    )
    return ExternalFunction(
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
