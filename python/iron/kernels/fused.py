# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Single-tile composition of the fused GEMM init, reduction and drain ABI."""

import hashlib
from pathlib import Path

import numpy as np
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import get_current_device
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    _EXTERN_CACHE,
    KernelContract,
    TensorLayout,
    _default_source_path,
    _detect_arch,
    _include_dirs,
)


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
) -> ExternalFunction:
    """Compute one bf16 ``A @ B`` tile, with an f32 reduction and fused epilogue.

    This bounded composition holds its operands and accumulator on one core;
    it is not the streaming whole-matrix operator. A and B use bf16 storage
    on both architectures (not the optional prepacked BFP16 B ABI).
    ``band_m`` and ``chunk_k`` subdivide the reduction; ``out_chunk`` subdivides
    the drain with a depth of two. The contract supplies the blocked storage
    layouts, including B's column-major ordering of row-major microblocks.

    ``epilogue`` is ``none``, ``gelu`` (the kernel's sigmoid approximation,
    not the tanh GELU curve), ``silu`` or ``sigmoid``. An optional finite
    ``(min, max)`` clamp follows the activation, before the bf16 conversion.
    """
    arch = _detect_arch()
    device = get_current_device(probe_runtime=False)
    if device is None:
        device = from_name("npu1" if arch == "aie2" else "npu2")
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

    def reference(a, b):
        c = a.reshape(-1, dim_m, dim_k).astype(np.float32) @ b.reshape(
            -1, dim_k, dim_n
        ).astype(np.float32)
        if epilogue != "none":
            x = c * 1.702 if epilogue == "gelu" else c
            sigmoid = (np.tanh(x * 0.5) + 1) * 0.5
            c = sigmoid if epilogue == "sigmoid" else c * sigmoid
        if clamp is not None:
            c = np.clip(c, clamp[0], clamp[1])
        return c.reshape(len(c), -1)

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
        "EPILOGUE_MODE": modes[epilogue],
    }
    compile_flags = ["-DROUND_CONV_EVEN"] + [
        f"-DMM_FUSED_{name}={value}" for name, value in flags.items()
    ]
    if clamp is not None:
        compile_flags += [
            "-DMM_FUSED_CLAMP=1",
            f"-DMM_FUSED_CLAMP_MIN={clamp[0]}f",
            f"-DMM_FUSED_CLAMP_MAX={clamp[1]}f",
        ]
    source = f'#include "{_default_source_path("mm_fused.cc", "generic")}"\n'
    include_dirs = _include_dirs()
    if arch == "aie2":
        from aie.utils import config

        runtime = Path(config.aie_runtime_lib_dir()) / "AIE2"
        include_dirs.append(str(runtime))
        source += f'#include "{runtime / "lut_based_ops.cpp"}"\n'
    source += """
extern "C" void fused_mm_tile(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  alignas(32) float acc[MM_FUSED_TILE_M * MM_FUSED_TILE_N];
  mm_fused_acc_init(acc);
  for (int k = 0; k < MM_FUSED_TILE_K / MM_FUSED_CT_K; ++k)
    for (int band = 0; band < MM_FUSED_TILE_M / MM_FUSED_TILE_MA; ++band)
      mm_fused_k_step(
          a + (k * MM_FUSED_TILE_M + band * MM_FUSED_TILE_MA) * MM_FUSED_CT_K,
          b + k * MM_FUSED_CT_K * MM_FUSED_TILE_N, acc, band);
  for (int outer = 0;
       outer < MM_FUSED_TILE_M * MM_FUSED_TILE_N / (2 * MM_FUSED_OUT_CHUNK);
       ++outer)
    for (int half = 0; half < 2; ++half)
      mm_fused_epilogue_chunk(
          c + (outer * 2 + half) * MM_FUSED_OUT_CHUNK, acc, outer, half);
}
"""
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
    )
    if key in _EXTERN_CACHE:
        return _EXTERN_CACHE[key]
    prefix = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    fn = ExternalFunction(
        "fused_mm_tile",
        source_string=source,
        arg_types=[
            np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]],
            np.ndarray[(dim_k * dim_n,), np.dtype[bfloat16]],
            np.ndarray[(dim_m * dim_n,), np.dtype[bfloat16]],
        ],
        include_dirs=include_dirs,
        compile_flags=compile_flags,
        # Object compilation renames every defined symbol, including the
        # included init/k_step/epilogue, zero kernels and AIE2 LUT exports.
        symbol_prefix=prefix,
    )
    fn.contract = KernelContract(
        roles=(In, In, Out),
        layouts=(
            TensorLayout((dim_m, dim_k), pack_a, unpack_a),
            TensorLayout((dim_k, dim_n), pack_b, unpack_b),
            TensorLayout((dim_m, dim_n), pack_c, unpack_c),
        ),
        reference=reference,
        tolerance=Tolerance.relative(
            0.02 if epilogue == "none" else 0.04,
            0.01 if epilogue == "none" else 0.04,
            note="bf16 store; activated path additionally narrows tanh to bf16",
        ),
        acc_dtype=np.float32,
        reduction=dim_k,
        ops_per_call=2 * dim_m * dim_k * dim_n,
        # The wrapper adds one explicit f32 accumulator to the target's
        # ordinary call stack. aiecc still checks the measured linked stack.
        stack_bytes=np.dtype(np.float32).itemsize * dim_m * dim_n
        + device.default_core_stack_bytes,
    )
    fn.dims = (dim_m, dim_k, dim_n)
    _EXTERN_CACHE[key] = fn
    return fn
