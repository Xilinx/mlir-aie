# cast_f32_bf16/cast_f32_bf16.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Element-wise f32 -> bf16 narrowing cast, IRON API + ``@iron.jit``.

NPU2-only: the underlying ``cast_f32_bf16.cc`` kernel lives under
``aie_kernels/aie2p/`` and has no aie2 counterpart.

Eight cores each cast ``n_vectors // 8`` vectors of ``vector_size`` elements.
Rounding is round-to-nearest-even.

Structurally this mirrors ``ml/norm``'s row-split ``@iron.jit`` design; the
difference here is the input and output tiles have different dtypes, which the
``transform_parallel`` / ``transform_parallel_binary`` algorithm helpers do not
support (they require uniform dtype across all tensors), hence the explicit
ObjectFifo/Worker wiring below rather than delegating to one of those.
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
    n_vectors: CompileTime[int] = 64,
    vector_size: CompileTime[int] = 4096,
):
    n_cores = 8
    vec = 16  # cast_f32_bf16_row<16>

    if n_vectors % n_cores != 0:
        raise ValueError(f"n_vectors ({n_vectors}) must be a multiple of {n_cores}")
    if vector_size % vec != 0:
        raise ValueError(f"vector_size ({vector_size}) must be a multiple of {vec}")

    rows_per_core = n_vectors // n_cores

    in_ty = np.ndarray[(n_vectors, vector_size), np.dtype[np.float32]]
    out_ty = np.ndarray[(n_vectors, vector_size), np.dtype[bfloat16]]
    chunk_in_ty = np.ndarray[(vector_size,), np.dtype[np.float32]]
    chunk_out_ty = np.ndarray[(vector_size,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_in_ty, name=f"in_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_out_ty, name=f"out_{i}") for i in range(n_cores)]

    cast_fn = _cast_extern(chunk_in_ty, chunk_out_ty)

    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out, vector_size)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), cast_fn])
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (n_vectors, vector_size), (rows_per_core, vector_size)
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
    p.add_argument("-s", "--n_vectors", type=int, default=64, help="number of vectors")
    p.add_argument(
        "-e", "--vector_size", type=int, default=4096, help="elements per vector"
    )
    return p


def _compile_kwargs(opts):
    return dict(n_vectors=opts.n_vectors, vector_size=opts.vector_size)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.n_vectors, opts.vector_size

    a_np = rng.uniform(-8.0, 8.0, size=(rows, cols)).astype(np.float32)
    a_t = iron.tensor(a_np, dtype=np.float32, device="npu")
    c_t = iron.zeros(rows * cols, dtype=bfloat16, device="npu")

    cast_f32_bf16(a_t, c_t, **_compile_kwargs(opts))

    # ml_dtypes rounds to nearest even, as the kernel's conv_even does, so this
    # is a bit-exact reference rather than a tolerance.
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
