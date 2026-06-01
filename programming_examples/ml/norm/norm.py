# norm/norm.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Row-wise bf16 norm (RMSNorm | LayerNorm) — Iron API + ``@iron.jit``.

NPU2-only: the underlying ``{rms,layer}_norm.cc`` kernels live under
``aie_kernels/aie2p/`` and have no aie2 counterpart.

Eight cores process ``sequence_length // 8`` rows each; one row =
``embedding_dim`` bf16 values. Per row:

  * rms (gamma=1, eps=1e-5):
      out = (x * gamma) / sqrt(mean(x^2) + eps)

  * layer (gamma=1, beta=0, eps=1e-5):
      out = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import device_from_args
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
}


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
    sequence_length: Compile[int] = 64,
    embedding_dim: Compile[int] = 4096,
    op: Compile[str] = "rms",
):
    device = iron.get_current_device()
    n_cores = 8

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )

    rows_per_core = sequence_length // n_cores
    total_volume = sequence_length * embedding_dim

    tensor_ty = np.ndarray[(total_volume,), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

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
        "-o", "--op", choices=("rms", "layer"), default="rms", help="norm flavor"
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


_VERIFY_CFG = {
    "rms": (_rms_norm_reference, 0.05),
    "layer": (_layer_norm_reference, 0.1),
}


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim
    a_np = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(bfloat16)
    c_np = np.zeros_like(a_np)

    a_t = iron.tensor(a_np.reshape(-1), dtype=bfloat16, device="npu")
    c_t = iron.tensor(c_np.reshape(-1), dtype=bfloat16, device="npu")

    norm(a_t, c_t, **_compile_kwargs(opts))

    ref_fn, atol = _VERIFY_CFG[opts.op]
    assert_pass(
        c_t.numpy().reshape(rows, cols),
        ref_fn(a_np),
        atol=atol,
        fail_msg=f"{opts.op}norm output mismatch",
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
