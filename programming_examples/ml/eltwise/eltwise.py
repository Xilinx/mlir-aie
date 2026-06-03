# eltwise/eltwise.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise bf16 binary op (add or mul) — IRON API + ``@iron.jit``.

Body delegates to ``iron.algorithms.transform_parallel_binary_typed``; the
per-tile kernel is selected by the ``op`` Compile knob and pulled from
``aie.iron.kernels`` (``kernels.add`` or ``kernels.mul``).
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import transform_parallel_binary_typed
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_FACTORIES = {"add": kernels.add, "mul": kernels.mul}
_NP_OPS = {"add": np.add, "mul": np.multiply}


@iron.jit
def eltwise(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    size: Compile[int] = 65536,
    num_channels: Compile[int] = 1,
    op: Compile[str] = "add",
):
    return transform_parallel_binary_typed(
        _KERNEL_FACTORIES[op](tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Eltwise")
    add_compile_args(p, with_elf=True)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    p.add_argument(
        "-ch", "--channels", type=int, default=1, help="channels per column (1 or 2)"
    )
    p.add_argument(
        "-o", "--op", choices=("add", "mul"), default="add", help="binary op"
    )
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, num_channels=opts.channels, op=opts.op)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1.0, 1.0, size=(opts.length,)).astype(bfloat16)
    b_np = rng.uniform(-1.0, 1.0, size=(opts.length,)).astype(bfloat16)
    c_np = np.zeros_like(a_np)

    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    b_t = iron.tensor(b_np, dtype=bfloat16, device="npu")
    c_t = iron.tensor(c_np, dtype=bfloat16, device="npu")

    eltwise(a_t, b_t, c_t, **_compile_kwargs(opts))

    expected = _NP_OPS[opts.op](a_np.astype(np.float32), b_np.astype(np.float32)).astype(
        bfloat16
    )
    assert_pass(
        c_t.numpy(),
        expected,
        atol=0.00390625,
        fail_msg=f"eltwise_{opts.op} output mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        eltwise,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
