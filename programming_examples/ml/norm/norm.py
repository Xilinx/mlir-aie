# norm/norm.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Row-wise norm (RMSNorm | LayerNorm) — IRON API + ``@iron.jit``.

NPU2-only: the underlying ``{rms,layer}_norm.cc`` kernels live under
``aie_kernels/aie2p/`` and have no aie2 counterpart.

Eight cores process ``sequence_length // 8`` rows each; one row =
``embedding_dim`` values. Per row:

  * rms (bf16, gamma=1, eps=1e-5):
      out = (x * gamma) / sqrt(mean(x^2) + eps)

  * layer (bf16, gamma=1, beta=0, eps=1e-5):
      out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

  * layer_f32: the same LayerNorm in f32 in/out, with a numerically stable
      centered two-pass variance (for the non-zero-mean inputs that f32 can
      represent and bf16 cannot).
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.utils.hostruntime.argparse import device_from_args
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_DIR = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p"
_KERNEL_SPEC = {
    "rms": ("rms_norm", _KERNEL_DIR / "rms_norm.cc"),
    "layer": ("layer_norm", _KERNEL_DIR / "layer_norm.cc"),
    "layer_f32": ("layer_norm_f32", _KERNEL_DIR / "layer_norm.cc"),
}

# rms / layer are bf16; layer_f32 is the f32-in/f32-out per-row LayerNorm.
_OP_DTYPE = {"rms": bfloat16, "layer": bfloat16, "layer_f32": np.float32}


def _norm_extern(op, chunk_type):
    sym, src = _KERNEL_SPEC[op]
    return ExternalFunction(
        sym,
        source_file=str(src),
        arg_types=[chunk_type, chunk_type, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def norm(
    a_in: In,
    c_out: Out,
    *,
    sequence_length: CompileTime[int] = 64,
    embedding_dim: CompileTime[int] = 4096,
    op: CompileTime[str] = "rms",
):
    device = iron.get_current_device()
    n_cores = 8
    vec = 16  # kernels reduce/store one aie::vector<T, 16> at a time

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )
    if embedding_dim % vec != 0:
        # The layer / layer_f32 kernels process full vec-wide chunks with no
        # scalar tail, so a non-multiple would silently drop the last columns.
        raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of {vec}")

    rows_per_core = sequence_length // n_cores

    dtype = _OP_DTYPE[op]
    tensor_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[dtype]]
    chunk_ty = np.ndarray[(embedding_dim,), np.dtype[dtype]]

    of_ins = [ObjectFifo(chunk_ty, name=f"in_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_ty, name=f"out_{i}") for i in range(n_cores)]

    norm_fn = _norm_extern(op, chunk_ty)

    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), norm_fn])
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, c):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_ins[i].prod(), a, taps[i])
        for i in range(n_cores):
            rt.drain(of_outs[i].cons(), c, taps[i], wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Norm")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=4096, help="cols per row")
    p.add_argument(
        "-o",
        "--op",
        choices=("rms", "layer", "layer_f32"),
        default="rms",
        help="norm flavor",
    )
    return p


def _compile_kwargs(opts):
    return dict(
        sequence_length=opts.sequence_length,
        embedding_dim=opts.embedding_dim,
        op=opts.op,
    )


def _rms_norm_reference(x_np):
    eps, gamma = 1e-5, 1.0
    x32 = x_np.astype(np.float32)
    rms = np.sqrt(np.sum(x32 * x32, axis=1) / x32.shape[1] + eps)
    return ((x32 * gamma) / rms[:, None]).astype(bfloat16)


def _layer_norm_reference(x_np):
    eps, gamma, beta = 1e-5, 1.0, 0.0
    x32 = x_np.astype(np.float32)
    mean = x32.mean(axis=1, keepdims=True)
    var = (x32 * x32).mean(axis=1, keepdims=True) - mean * mean
    inv_std = 1.0 / np.sqrt(var + eps)
    return ((x32 - mean) * inv_std * gamma + beta).astype(bfloat16)


def _layer_norm_f32_reference(x_np):
    # Gold reference for the f32 kernel: centered two-pass variance in f64, f32
    # output. Centered (not E[x^2]-mean^2) so it stays exact on the non-zero-mean
    # input the kernel is exercised with.
    eps, gamma, beta = 1e-5, 1.0, 0.0
    x64 = x_np.astype(np.float64)
    mean = x64.mean(axis=1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return ((x64 - mean) * inv_std * gamma + beta).astype(np.float32)


# per op: (reference fn, elementwise atol, rtol, mean per-row rel-L2 ceiling).
# rtol: assert_pass defaults to 0.128 (the bf16/LUT-tuned relative tolerance)
# when only atol is given, which would dominate an O(1) LayerNorm output and mask
# the f32 kernel's real error, so layer_f32 pins rtol=0 to let its tight absolute
# atol actually govern. The bf16 ops keep the default (None).
# rel-L2: aggregate accumulation guard, bf16 ops only; the f32 path is checked
# directly by its atol.
_VERIFY_CFG = {
    "rms": (_rms_norm_reference, 0.05, None, None),
    "layer": (_layer_norm_reference, 0.05, None, 0.01),
    "layer_f32": (_layer_norm_f32_reference, 1e-3, 0.0, None),
}


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim
    dtype = _OP_DTYPE[opts.op]

    if opts.op == "layer_f32":
        # Non-zero row mean (~100): representable in f32 but not bf16, and large
        # enough that E[x^2]-mean^2 loses the variance to f32 cancellation while
        # the centered two-pass stays exact. Exercises what the f32 path is for.
        a_np = (rng.uniform(-1.0, 1.0, size=(rows, cols)) + 100.0).astype(dtype)
    else:
        a_np = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)

    a_t = iron.tensor(a_np, dtype=dtype, device="npu")
    c_t = iron.zeros_like(a_t)

    norm(a_t, c_t, **_compile_kwargs(opts))

    ref_fn, atol, rtol, rel_l2_max = _VERIFY_CFG[opts.op]
    out = c_t.numpy().reshape(rows, cols)
    ref = ref_fn(a_np)
    assert_pass(
        out, ref, atol=atol, rtol=rtol, fail_msg=f"{opts.op}norm output mismatch"
    )

    # Aggregate regression guard. The elementwise atol above is bounded by bf16
    # output quantization and barely moves when the reduction loses precision,
    # so it cannot catch an accumulation regression on its own; the mean per-row
    # rel-L2 can (f32 sum accumulation ~0.6%, a bf16 sum ~1.1-1.9%).
    if rel_l2_max is not None:
        o = out.astype(np.float32)
        r = ref.astype(np.float32)
        rel_l2 = np.sqrt(((o - r) ** 2).sum(axis=1) / (r**2).sum(axis=1)).mean()
        assert rel_l2 <= rel_l2_max, (
            f"{opts.op}norm mean per-row rel-L2 {rel_l2:.4f} exceeds "
            f"{rel_l2_max} (accumulation-precision regression?)"
        )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        norm,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
