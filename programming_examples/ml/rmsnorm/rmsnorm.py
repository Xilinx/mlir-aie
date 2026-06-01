# rmsnorm/rmsnorm.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Row-wise bf16 RMSNorm — Iron API design with ``@iron.jit`` compilation.

NPU2-only: the underlying ``rms_norm.cc`` kernel lives under
``aie_kernels/aie2p/`` and has no aie2 counterpart.

Eight cores process ``sequence_length // 8`` rows each; one row =
``embedding_dim`` bf16 values.  Per row:
    rms = sqrt(mean(x²) + eps), out = (x * gamma) / rms.
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

_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/rms_norm.cc"


def _rms_norm_extern(chunk_type):
    return ExternalFunction(
        "rms_norm",
        source_file=str(_KERNEL_SRC),
        arg_types=[chunk_type, chunk_type, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def rmsnorm(
    a_in: In,
    c_out: Out,
    *,
    sequence_length: Compile[int] = 64,
    embedding_dim: Compile[int] = 4096,
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

    rms_norm_fn = _rms_norm_extern(chunk_ty)

    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), rms_norm_fn])
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
    p = argparse.ArgumentParser(prog="AIE RMSNorm")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=4096, help="cols per row")
    return p


def _compile_kwargs(opts):
    return dict(
        sequence_length=opts.sequence_length,
        embedding_dim=opts.embedding_dim,
    )


def _rms_norm_reference(x_np: np.ndarray) -> np.ndarray:
    eps = 1e-5
    gamma = 1.0
    x32 = x_np.astype(np.float32)
    sum_sq = np.sum(x32 * x32, axis=1)
    rms = np.sqrt(sum_sq / x32.shape[1] + eps)
    out = (x32 * gamma) / rms[:, None]
    return out.astype(bfloat16)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim
    a_np = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(bfloat16)
    c_np = np.zeros_like(a_np)

    a_t = iron.tensor(a_np.reshape(-1), dtype=bfloat16, device="npu")
    c_t = iron.tensor(c_np.reshape(-1), dtype=bfloat16, device="npu")

    rmsnorm(a_t, c_t, **_compile_kwargs(opts))

    expected = _rms_norm_reference(a_np)
    assert_pass(
        c_t.numpy().reshape(rows, cols),
        expected,
        atol=0.05,
        fail_msg="rmsnorm output mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        rmsnorm,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
