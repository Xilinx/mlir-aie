# eltwise_mul/eltwise_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 multiply — Iron API design with ``@iron.jit`` compilation.

Saturates DDR bandwidth by using every column's shim DMA in/out pairs.
The per-tile kernel comes from ``aie.iron.kernels.mul`` (auto-built by
``compile_mlir_module(device=...)`` into the JIT work_dir).

Two invocation modes:

  * standalone:   ``python3 eltwise_mul.py``
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
def eltwise_mul(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    size: Compile[int] = 65536,
    num_channels: Compile[int] = 1,
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

    of_in1s = [
        ObjectFifo(line_type, name=f"in1_{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_in2s = [
        ObjectFifo(line_type, name=f"in2_{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out_{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    mul_fn = kernels.mul(tile_size=line_size)

    def core_fn(of_in1, of_in2, of_out, kernel):
        elem_out = of_out.acquire(1)
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        kernel(elem_in1, elem_in2, elem_out)
        of_in1.release(1)
        of_in2.release(1)
        of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [
                of_in1s[i * num_channels + j].cons(),
                of_in2s[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                mul_fn,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    taps = TensorTiler2D.simple_tiler((1, size), (1, chunk))

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type, transfer_type) as (a, b, c):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_columns):
            for j in range(num_channels):
                idx = i * num_channels + j
                rt.fill(of_in1s[idx].prod(), a, taps[idx], task_group=tg)
                rt.fill(of_in2s[idx].prod(), b, taps[idx], task_group=tg)
        for i in range(num_columns):
            for j in range(num_channels):
                idx = i * num_channels + j
                rt.drain(
                    of_outs[idx].cons(),
                    c,
                    taps[idx],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    return Program(device, rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Eltwise Mul")
    add_compile_args(p)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    p.add_argument(
        "-ch", "--channels", type=int, default=1, help="channels per column (1 or 2)"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_channels=opts.channels)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1.0, 1.0, size=(opts.length,)).astype(bfloat16)
    b_np = rng.uniform(-1.0, 1.0, size=(opts.length,)).astype(bfloat16)
    c_np = np.zeros_like(a_np)

    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    b_t = iron.tensor(b_np, dtype=bfloat16, device="npu")
    c_t = iron.tensor(c_np, dtype=bfloat16, device="npu")

    eltwise_mul(a_t, b_t, c_t, **_compile_kwargs(opts))

    expected = (a_np.astype(np.float32) * b_np.astype(np.float32)).astype(bfloat16)
    assert_pass(
        c_t.numpy(),
        expected,
        atol=0.00390625,
        fail_msg="eltwise_mul output mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        eltwise_mul,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: from_name(o.dev, n_cols=None),
    )


if __name__ == "__main__":
    main()
