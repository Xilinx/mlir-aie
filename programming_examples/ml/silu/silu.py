# silu/silu.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 SILU — Iron API design with ``@iron.jit`` compilation.

Body delegates to ``iron.algorithms.transform_parallel_typed`` with
``num_channels=2``; the per-tile kernel comes from ``aie.iron.kernels.silu``.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import transform_parallel_typed
from aie.iron.device import device_from_args
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
    return transform_parallel_typed(
        kernels.silu(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE SILU")
    add_compile_args(p, with_elf=True)
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
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
