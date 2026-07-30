# cast_f32_bf16/cast_f32_bf16.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Element-wise f32 -> bf16 narrowing cast, IRON API + ``@iron.jit``.

NPU2-only: the underlying ``cast_f32_bf16.cc`` kernel lives under
``aie_kernels/aie2p/`` and has no aie2 counterpart.

Eight cores each cast ``sequence_length // 8`` rows; one row is
``embedding_dim`` f32 values in, ``embedding_dim`` bf16 values out.
Rounding is round-to-nearest-even, matching a host f32->bf16 pack that also
rounds to nearest even, so an on-chip and a host cast of the same input agree
bit-for-bit. Structurally this mirrors ``ml/norm``'s row-split ``@iron.jit``
design (same 8-core-per-row shape); the difference here is the input and
output tiles have different dtypes, which the ``transform_parallel`` /
``transform_parallel_binary`` algorithm helpers do not support (they require
uniform dtype across all tensors), hence the explicit ObjectFifo/Worker
wiring below rather than delegating to one of those.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.hostruntime.argparse import device_from_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_DIR = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p"


def _cast_extern(chunk_in_ty, chunk_out_ty):
    return ExternalFunction(
        "cast_f32_bf16_row",
        source_file=str(_KERNEL_DIR / "cast_f32_bf16.cc"),
        arg_types=[chunk_in_ty, chunk_out_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def cast_f32_bf16(
    a_in: In,
    c_out: Out,
    *,
    sequence_length: CompileTime[int] = 64,
    embedding_dim: CompileTime[int] = 4096,
):
    n_cores = 8
    vec = 16  # cast_f32_bf16_row<16> vectorizes cols by 16

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )
    if embedding_dim % vec != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of {vec}")

    rows_per_core = sequence_length // n_cores

    in_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[np.float32]]
    out_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[bfloat16]]
    chunk_in_ty = np.ndarray[(embedding_dim,), np.dtype[np.float32]]
    chunk_out_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_in_ty, name=f"in_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_out_ty, name=f"out_{i}") for i in range(n_cores)]

    cast_fn = _cast_extern(chunk_in_ty, chunk_out_ty)

    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), cast_fn])
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
            in_ty,
            out_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )

    device = iron.get_current_device()
    return Program(device, rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Cast f32->bf16")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=4096, help="cols per row")
    return p


def _compile_kwargs(opts):
    return dict(sequence_length=opts.sequence_length, embedding_dim=opts.embedding_dim)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim

    a_np = rng.uniform(-8.0, 8.0, size=(rows, cols)).astype(np.float32)
    a_t = iron.tensor(a_np, dtype=np.float32, device="npu")
    c_t = iron.zeros(rows * cols, dtype=bfloat16, device="npu")

    cast_f32_bf16(a_t, c_t, **_compile_kwargs(opts))

    # ml_dtypes.bfloat16's f32->bf16 cast rounds to nearest even, matching the
    # kernel's aie::rounding_mode::conv_even, a bit-exact reference.
    expected = a_np.astype(bfloat16)
    out = c_t.numpy().reshape(rows, cols)
    assert_pass(out, expected, fail_msg="cast_f32_bf16 output mismatch")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        cast_f32_bf16,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
