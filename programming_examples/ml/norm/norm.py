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

  * layer_affine_cast: the same centered LayerNorm, with a REAL per-column
      gamma/beta (not fixed to 1/0) and a f32 -> bf16 narrowing cast fused
      into the same dispatch:
        out = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta -> bf16
      gamma/beta are the same for every row of a call, packed into one
      ``[2 * embedding_dim]`` buffer (gamma then beta) that every core loads
      once, before its row loop, rather than per row. This op has a
      different tensor shape than the other three (f32 in, bf16 out, plus
      the gamma/beta parameter tensor), so it is wired by a second design,
      ``norm_affine``, below; see its docstring for why.
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass
from ml_dtypes import bfloat16

_KERNEL_DIR = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p"
_KERNEL_SPEC = {
    "rms": ("rms_norm", _KERNEL_DIR / "rms_norm.cc"),
    "layer": ("layer_norm", _KERNEL_DIR / "layer_norm.cc"),
    "layer_f32": ("layer_norm_f32", _KERNEL_DIR / "layer_norm.cc"),
    "layer_affine_cast": ("layer_norm_affine_cast", _KERNEL_DIR / "layer_norm.cc"),
}

# rms / layer are bf16; layer_f32 is the f32-in/f32-out per-row LayerNorm.
# layer_affine_cast is handled by the separate norm_affine design below (f32
# in, bf16 out, plus a gamma/beta parameter tensor), so it carries no entry
# here.
_OP_DTYPE = {"rms": bfloat16, "layer": bfloat16, "layer_f32": np.float32}

# Ops wired through norm_affine (a real per-column gamma/beta plus an output
# cast) rather than through norm (same dtype in/out, gamma=1/beta=0 fixed).
_AFFINE_OPS = frozenset({"layer_affine_cast"})


def _norm_extern(op, chunk_type):
    sym, src = _KERNEL_SPEC[op]
    return ExternalFunction(
        sym,
        source_file=str(src),
        arg_types=[
            chunk_type,
            chunk_type,
            np.int32,  # pyright: ignore[reportArgumentType]
        ],
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

    # layer_norm_f32's frame is 1024 bytes on its own (llvm-readelf
    # --stack-sizes), which is exactly the target default, so the core
    # overruns the reservation into the buffers the linker places above it.
    workers = [
        Worker(
            core_fn, [of_ins[i].cons(), of_outs[i].prod(), norm_fn], stack_size=2048
        )
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    def sequence(a, c, in_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(a, taps[i])
        for i in range(n_cores):
            out_conses[i].drain(c, taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            tensor_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _norm_affine_extern(op, chunk_in_ty, chunk_gb_ty, chunk_out_ty):
    sym, src = _KERNEL_SPEC[op]
    return ExternalFunction(
        sym,
        source_file=str(src),
        arg_types=[
            chunk_in_ty,
            chunk_gb_ty,
            chunk_out_ty,
            np.int32,  # pyright: ignore[reportArgumentType]
        ],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def norm_affine(
    a_in: In,
    gb_in: In,
    c_out: Out,
    *,
    sequence_length: CompileTime[int] = 64,
    embedding_dim: CompileTime[int] = 1024,
    op: CompileTime[str] = "layer_affine_cast",
):
    """Row-wise norm with a real per-column affine, fused with an output cast.

    Structurally mirrors ``norm`` above (same 8-core-per-row shape), extended
    two ways a same-dtype, two-tensor design cannot express: the input and
    output tiles have different dtypes (f32 in, bf16 out, as in
    ``ml/cast_f32_bf16``), and there is a third, per-core-constant parameter
    tensor (gamma/beta) alongside the per-row data tensor. Neither fits
    ``norm``'s single ``chunk_ty``/two-``ObjectFifo`` wiring, hence the
    separate design rather than a branch inside ``norm`` itself.
    """
    device = iron.get_current_device()
    n_cores = 8
    vec = 16  # layer_norm_affine_cast vectorizes cols by 16

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )
    if embedding_dim % vec != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of {vec}")

    rows_per_core = sequence_length // n_cores

    in_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[np.float32]]
    gb_ty = np.ndarray[(1, 2 * embedding_dim), np.dtype[np.float32]]
    out_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[bfloat16]]

    chunk_in_ty = np.ndarray[(embedding_dim,), np.dtype[np.float32]]
    chunk_gb_ty = np.ndarray[(2 * embedding_dim,), np.dtype[np.float32]]
    chunk_out_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_in_ty, name=f"in_{i}") for i in range(n_cores)]
    # depth=1: gamma/beta are constant for the whole row loop (acquired once,
    # never released mid-loop below), so there is nothing to double-buffer,
    # and at depth=2 the [2 * embedding_dim] f32 buffer alone can exceed a
    # tile's 64 KB L1 budget once the row in/out buffers are added.
    of_gbs = [ObjectFifo(chunk_gb_ty, name=f"gb_{i}", depth=1) for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_out_ty, name=f"out_{i}") for i in range(n_cores)]

    norm_fn = _norm_affine_extern(op, chunk_in_ty, chunk_gb_ty, chunk_out_ty)

    def core_fn(of_in, of_gb, of_out, kernel):
        # gamma/beta are constant for every row this core sees: acquire once,
        # hold for the whole row loop, release after the last row.
        elem_gb = of_gb.acquire(1)
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_gb, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)
        of_gb.release(1)

    workers = [
        Worker(
            core_fn,
            [of_ins[i].cons(), of_gbs[i].cons(), of_outs[i].prod(), norm_fn],
        )
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )
    # One tile == the whole gb tensor; every core loads the SAME full range.
    gb_taps = TensorTiler2D.simple_tiler((1, 2 * embedding_dim))

    def sequence(a, gb, c, in_prods, gb_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(a, taps[i])
        for i in range(n_cores):
            gb_prods[i].fill(gb, gb_taps[0])
        for i in range(n_cores):
            out_conses[i].drain(c, taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            in_ty,
            gb_ty,
            out_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_gbs[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )

    return Program(device, rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Norm")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=4096, help="cols per row")
    p.add_argument(
        "-o",
        "--op",
        choices=("rms", "layer", "layer_f32", "layer_affine_cast"),
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


def _layer_affine_cast_reference(x_np, gamma_np, beta_np):
    eps = 1e-5
    x32 = x_np.astype(np.float32)
    mean = x32.mean(axis=1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    y = (x32 - mean) * inv_std * gamma_np + beta_np
    return y.astype(bfloat16)


# per op: (reference fn, elementwise atol, rtol, mean per-row rel-L2 ceiling).
# rtol: assert_pass defaults to 0.128 (the bf16/LUT-tuned relative tolerance)
# when only atol is given, which would dominate an O(1) LayerNorm output and mask
# the f32 kernel's real error, so layer_f32 pins rtol=0 to let its tight absolute
# atol actually govern. The bf16 ops keep the default (None).
# rel-L2: aggregate accumulation guard, bf16-output ops only; the f32 path is
# checked directly by its atol. layer_affine_cast reuses layer's gate: same
# centered two-pass reduction and the same bf16 output quantization, just with
# a real affine and an f32 input instead of a fixed identity affine and a bf16
# one, there is no reason to expect a different accumulation-error regime.
_VERIFY_CFG = {
    "rms": (_rms_norm_reference, 0.05, None, None),
    "layer": (_layer_norm_reference, 0.05, None, 0.01),
    "layer_f32": (_layer_norm_f32_reference, 1e-3, 0.0, None),
    "layer_affine_cast": (_layer_affine_cast_reference, 0.05, None, 0.01),
}


def _run_and_verify_norm(opts):
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
    _check(out, ref, opts.op, atol, rtol, rel_l2_max)


def _run_and_verify_norm_affine(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim

    a_np = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)
    gamma_np = rng.uniform(0.5, 1.5, size=(cols,)).astype(np.float32)
    beta_np = rng.uniform(-0.5, 0.5, size=(cols,)).astype(np.float32)
    gb_np = np.concatenate([gamma_np, beta_np]).reshape(1, 2 * cols)

    a_t = iron.tensor(a_np, dtype=np.float32, device="npu")
    gb_t = iron.tensor(gb_np, dtype=np.float32, device="npu")
    c_t = iron.zeros(rows * cols, dtype=bfloat16, device="npu")

    norm_affine(a_t, gb_t, c_t, **_compile_kwargs(opts))

    ref_fn, atol, rtol, rel_l2_max = _VERIFY_CFG[opts.op]
    out = c_t.numpy().reshape(rows, cols)
    ref = ref_fn(a_np, gamma_np, beta_np)
    _check(out, ref, opts.op, atol, rtol, rel_l2_max)


def _check(out, ref, op, atol, rtol, rel_l2_max):
    assert_pass(out, ref, atol=atol, rtol=rtol, fail_msg=f"{op}norm output mismatch")

    # Aggregate regression guard. The elementwise atol above is bounded by bf16
    # output quantization and barely moves when the reduction loses precision,
    # so it cannot catch an accumulation regression on its own; the mean per-row
    # rel-L2 can (f32 sum accumulation ~0.6%, a bf16 sum ~1.1-1.9%).
    if rel_l2_max is not None:
        o = out.astype(np.float32)
        r = ref.astype(np.float32)
        rel_l2 = np.sqrt(((o - r) ** 2).sum(axis=1) / (r**2).sum(axis=1)).mean()
        assert rel_l2 <= rel_l2_max, (
            f"{op}norm mean per-row rel-L2 {rel_l2:.4f} exceeds "
            f"{rel_l2_max} (accumulation-precision regression?)"
        )


def _run_and_verify(opts):
    if opts.op in _AFFINE_OPS:
        _run_and_verify_norm_affine(opts)
    else:
        _run_and_verify_norm(opts)


def main():
    opts = _make_argparser().parse_args()
    design = norm_affine if opts.op in _AFFINE_OPS else norm
    run_design_cli(
        design,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
