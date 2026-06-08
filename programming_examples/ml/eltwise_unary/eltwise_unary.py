# eltwise_unary/eltwise_unary.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 unary op (ReLU | SiLU | GELU) — IRON API + ``@iron.jit``.

Body delegates to ``iron.algorithms.transform_parallel_typed`` with
``num_channels=2`` so both shim DMA channels per column are driven; the
per-tile kernel is selected by the ``op`` Compile knob and pulled from
``aie.iron.kernels``.
"""

import argparse
import math

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

_KERNEL_FACTORIES = {"relu": kernels.relu, "silu": kernels.silu, "gelu": kernels.gelu}


def _relu_ref_f32(x):
    return np.maximum(x.astype(np.float32), 0.0)


def _silu_ref_f32(x):
    x = x.astype(np.float32)
    return x / (1.0 + np.exp(-x))


def _gelu_ref_f32(x):
    x = x.astype(np.float32)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# Tolerances picked per-op to match the LUT-backed kernel: relu is exact,
# silu/gelu are LUT approximations.
_VERIFY_CFG = {
    "relu": (_relu_ref_f32, dict(fail_msg="relu output does not match max(in, 0)")),
    "silu": (
        _silu_ref_f32,
        dict(rtol=0.128, fail_msg="silu output outside LUT tolerance"),
    ),
    "gelu": (
        _gelu_ref_f32,
        dict(
            rtol=0.128,
            atol=0.05,
            fail_msg="gelu output outside LUT tolerance",
        ),
    ),
}


@iron.jit
def eltwise_unary(
    a_in: In,
    b_out: Out,
    *,
    size: Compile[int] = 65536,
    num_channels: Compile[int] = 2,
    op: Compile[str] = "relu",
):
    return transform_parallel_typed(
        _KERNEL_FACTORIES[op](tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Eltwise Unary")
    add_compile_args(p, with_elf=True)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    p.add_argument(
        "-ch", "--channels", type=int, default=2, help="channels per column (1 or 2)"
    )
    p.add_argument(
        "-o", "--op", choices=("relu", "silu", "gelu"), default="relu", help="unary op"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_channels=opts.channels, op=opts.op)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.uniform(-3.0, 3.0, size=(opts.length,)).astype(bfloat16)
    a_t = iron.tensor(in_np, dtype=bfloat16, device="npu")
    b_t = iron.zeros_like(a_t)

    eltwise_unary(a_t, b_t, **_compile_kwargs(opts))

    ref_fn, verify_kwargs = _VERIFY_CFG[opts.op]
    assert_pass(b_t.numpy(), ref_fn(in_np), **verify_kwargs)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        eltwise_unary,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
