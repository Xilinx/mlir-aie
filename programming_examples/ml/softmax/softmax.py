# softmax/softmax.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Tile-wise bf16 softmax — IRON API design with ``@iron.jit`` compilation.

Softmax is computed independently per 1024-element tile (no cross-tile
reduction), so the design scales the same way as ml/eltwise_unary: body
delegates to ``iron.algorithms.transform_parallel_typed`` with
``num_channels=2`` and ``pass_size_to_kernel=True`` (the softmax kernel
signature is ``(in, out, line_size)``).
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import transform_parallel_typed
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def softmax(
    a_in: In,
    b_out: Out,
    *,
    size: Compile[int] = 262144,
    num_channels: Compile[int] = 2,
):
    return transform_parallel_typed(
        kernels.softmax(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=True,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Softmax")
    add_compile_args(p, with_elf=True)
    p.add_argument("-l", "--length", type=int, default=262144, help="elements")
    p.add_argument(
        "-ch", "--channels", type=int, default=2, help="channels per column (1 or 2)"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_channels=opts.channels)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.uniform(-4.0, 8.0, size=(opts.length,)).astype(bfloat16)
    a_t = iron.tensor(in_np, dtype=bfloat16, device="npu")
    b_t = iron.zeros_like(a_t)

    softmax(a_t, b_t, **_compile_kwargs(opts))

    expected = kernels.softmax_ref(in_np, tile_size=1024)
    assert_pass(
        b_t.numpy(),
        expected,
        rtol=0.128,
        fail_msg="softmax output does not match reference",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        softmax,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
