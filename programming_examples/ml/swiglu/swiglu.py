# swiglu/swiglu.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 SwiGLU — IRON API design with ``@iron.jit`` compilation.

SwiGLU(x, w1, w2) = (x * w1) * silu(x * w2), computed per 1024-element line.
Weights are interleaved (1024 of w1, then 1024 of w2, ...) into a single
2*size weight buffer, matching the host-side packing in ``test.cpp``.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.device import device_from_args
from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def swiglu(
    a_in: In,
    w_in: In,
    b_out: Out,
    *,
    size: Compile[int] = 16384,
    num_columns: Compile[int] = 4,
):
    xfr_dtype = bfloat16
    device = iron.get_current_device()

    if num_columns > device.cols:
        raise ValueError(
            f"num_columns ({num_columns}) exceeds device.cols ({device.cols})"
        )
    if (size % 1024) % num_columns != 0:
        raise ValueError(
            f"size ({size}) must be a multiple of 1024 and divisible by "
            f"{num_columns} columns"
        )

    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]
    transfer_type_wts = np.ndarray[(2 * size,), np.dtype[xfr_dtype]]
    chunk = size // num_columns

    of_ins = [ObjectFifo(line_type, name=f"in{i}") for i in range(num_columns)]
    of_wts = [ObjectFifo(line_type, depth=4, name=f"w{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(line_type, name=f"out{i}") for i in range(num_columns)]

    swiglu_fn = kernels.swiglu(tile_size=line_size)

    def core_fn(of_in, of_wts, of_out, kernel):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        elem_wts = of_wts.acquire(2)
        kernel(elem_in, elem_wts[0], elem_wts[1], elem_out)
        of_wts.release(2)
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [
                of_ins[i].cons(),
                of_wts[i].cons(),
                of_outs[i].prod(),
                swiglu_fn,
            ],
        )
        for i in range(num_columns)
    ]

    taps = TensorTiler2D.simple_tiler((1, size), (1, chunk))
    taps_wts = TensorTiler2D.simple_tiler((1, 2 * size), (1, 2 * chunk))

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type_wts, transfer_type) as (a, w, b):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_columns):
            rt.fill(of_ins[i].prod(), a, taps[i], task_group=tg)
            rt.fill(of_wts[i].prod(), w, taps_wts[i], task_group=tg)
        for i in range(num_columns):
            rt.drain(of_outs[i].cons(), b, taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE SwiGLU")
    add_compile_args(p)
    p.add_argument("-l", "--length", type=int, default=16384, help="elements")
    p.add_argument(
        "-co", "--columns", type=int, default=4, help="number of columns to use"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_columns=opts.columns)


def _silu_ref_f32(x):
    return x / (1.0 + np.exp(-x))


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    n = opts.length
    in_np = rng.uniform(-1.0, 1.0, size=(n,)).astype(bfloat16)
    w1_np = rng.uniform(-1.0, 1.0, size=(n,)).astype(bfloat16)
    w2_np = rng.uniform(-1.0, 1.0, size=(n,)).astype(bfloat16)

    # Host interleaves: [w1[0:1024], w2[0:1024], w1[1024:2048], w2[1024:2048], ...]
    w_np = np.empty((2 * n,), dtype=bfloat16)
    for i in range(0, n, 1024):
        w_np[2 * i : 2 * i + 1024] = w1_np[i : i + 1024]
        w_np[2 * i + 1024 : 2 * i + 2048] = w2_np[i : i + 1024]

    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np, dtype=bfloat16, device="npu")
    w_t = iron.tensor(w_np, dtype=bfloat16, device="npu")
    b_t = iron.tensor(out_np, dtype=bfloat16, device="npu")

    swiglu(a_t, w_t, b_t, **_compile_kwargs(opts))

    x_f32 = in_np.astype(np.float32)
    w1_f32 = w1_np.astype(np.float32)
    w2_f32 = w2_np.astype(np.float32)
    expected = ((x_f32 * w1_f32) * _silu_ref_f32(x_f32 * w2_f32)).astype(bfloat16)
    assert_pass(
        b_t.numpy(),
        expected,
        rtol=0.128,
        fail_msg="swiglu output does not match reference",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        swiglu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
