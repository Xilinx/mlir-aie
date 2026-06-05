# rope/rope.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Row-wise bf16 RoPE (Rotary Position Embedding) — IRON API design.

NPU2-only: ``rope.cc`` lives under ``aie_kernels/aie2p/`` only.

Four cores process ``sequence_length // 4`` rows each.  Per row, on
even/odd element pairs::

    out[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
    out[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)

The cos/sin LUT (interleaved as cos, sin, cos, sin, ...) is generated
host-side from ``theta = 10000`` per the canonical RoPE formula.
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

_KERNEL_SRC = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/rope.cc"


def _rope_extern(chunk_type):
    return ExternalFunction(
        "rope",
        source_file=str(_KERNEL_SRC),
        arg_types=[chunk_type, chunk_type, chunk_type, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def rope(
    a_in: In,
    lut_in: In,
    c_out: Out,
    *,
    sequence_length: Compile[int] = 64,
    embedding_dim: Compile[int] = 4096,
):
    device = iron.get_current_device()
    n_cores = 4

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be even")

    rows_per_core = sequence_length // n_cores

    tensor_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_ty, name=f"in_{i}") for i in range(n_cores)]
    of_luts = [ObjectFifo(chunk_ty, name=f"lut_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_ty, name=f"out_{i}") for i in range(n_cores)]

    rope_fn = _rope_extern(chunk_ty)

    def core_fn(of_in, of_lut, of_out, kernel):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_lut = of_lut.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_lut, elem_out, embedding_dim)
            of_in.release(1)
            of_lut.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [of_ins[i].cons(), of_luts[i].cons(), of_outs[i].prod(), rope_fn],
        )
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a, lut, c):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_ins[i].prod(), a, taps[i])
            rt.fill(of_luts[i].prod(), lut, taps[i])
        for i in range(n_cores):
            rt.drain(of_outs[i].cons(), c, taps[i], wait=True)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE RoPE")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=4096, help="cols per row")
    return p


def _compile_kwargs(opts):
    return dict(
        sequence_length=opts.sequence_length,
        embedding_dim=opts.embedding_dim,
    )


def _build_rope_lut(rows: int, cols: int, theta: float = 10000.0) -> np.ndarray:
    """Generate the (cos, sin) interleaved LUT matching test.cpp's init."""
    dims = cols // 2
    inv_freq = 1.0 / (theta ** (np.arange(dims, dtype=np.float32) * 2.0 / cols))
    r = np.arange(rows, dtype=np.float32)[:, None]
    angles = r * inv_freq[None, :]
    cos_v = np.cos(angles)
    sin_v = np.sin(angles)
    lut = np.empty((rows, cols), dtype=np.float32)
    lut[:, 0::2] = cos_v
    lut[:, 1::2] = sin_v
    return lut.astype(bfloat16)


def _rope_reference(x_np: np.ndarray, lut_np: np.ndarray) -> np.ndarray:
    rows, cols = x_np.shape
    x32 = x_np.astype(np.float32)
    lut32 = lut_np.astype(np.float32)
    cos_v = lut32[:, 0::2]
    sin_v = lut32[:, 1::2]
    x_even = x32[:, 0::2]
    x_odd = x32[:, 1::2]
    out = np.empty_like(x32)
    out[:, 0::2] = x_even * cos_v - x_odd * sin_v
    out[:, 1::2] = x_even * sin_v + x_odd * cos_v
    return out.astype(bfloat16)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim
    # Match test.cpp input range: random bf16 in [-4, 8].
    a_np = rng.uniform(-4.0, 8.0, size=(rows, cols)).astype(bfloat16)
    lut_np = _build_rope_lut(rows, cols)
    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    lut_t = iron.tensor(lut_np, dtype=bfloat16, device="npu")
    c_t = iron.zeros_like(a_t)

    rope(a_t, lut_t, c_t, **_compile_kwargs(opts))

    expected = _rope_reference(a_np, lut_np)
    # atol=0.1 (vs test.cpp's 0.05): with random inputs in [-4, 8] and
    # cos/sin in [-1, 1], the worst-case bf16 quantization on the
    # x*cos - x*sin sum lands at ~0.06 in a handful of cells.
    assert_pass(
        c_t.numpy().reshape(rows, cols),
        expected,
        atol=0.1,
        fail_msg="rope output mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        rope,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
