# silu/silu.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 SILU — Iron API design with ``@iron.jit`` compilation.

Same shape as ml/relu: every shim DMA in/out pair feeds a ``kernels.silu``
worker that LUT-approximates ``x * sigmoid(x)`` (SILU / Swish) on its ``1024``-element
sub-vector.

Two invocation modes:

  * standalone:   ``python3 silu.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH [--elf-path=PATH]``
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.device import from_name
from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def silu(
    a_in: In,
    b_out: Out,
    *,
    size: Compile[int] = 65536,
    num_channels: Compile[int] = 2,
):
    xfr_dtype = bfloat16
    device = iron.get_current_device()
    num_columns = device.cols

    if num_channels not in (1, 2):
        raise ValueError(f"num_channels must be 1 or 2, got {num_channels}")
    if (size % 1024) % num_columns % num_channels != 0:
        raise ValueError(
            f"size ({size}) must be a multiple of 1024 and divisible by "
            f"{num_columns} columns × {num_channels} channels"
        )

    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]
    chunk = size // num_columns // num_channels

    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    silu_fn = kernels.silu(tile_size=line_size)

    def core_fn(of_in, of_out, kernel):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        kernel(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                silu_fn,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    taps = TensorTiler2D.simple_tiler((1, size), (1, chunk))

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a, b):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE SILU")
    add_compile_args(p)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    p.add_argument(
        "-ch", "--channels", type=int, default=2, help="channels per column (1 or 2)"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_channels=opts.channels)


def _silu_ref_f32(x):
    """SILU (a.k.a. swish): x * sigmoid(x) = x / (1 + exp(-x))."""
    return x.astype(np.float32) / (1.0 + np.exp(-x.astype(np.float32)))


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.uniform(-3.0, 3.0, size=(opts.length,)).astype(bfloat16)
    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np, dtype=bfloat16, device="npu")
    b_t = iron.tensor(out_np, dtype=bfloat16, device="npu")

    silu(a_t, b_t, **_compile_kwargs(opts))

    expected = _silu_ref_f32(in_np)
    assert_pass(
        b_t.numpy(),
        expected,
        rtol=0.128,
        fail_msg="silu output outside LUT tolerance",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        silu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=None),
    )


if __name__ == "__main__":
    main()
