# vector_scalar_add/vector_scalar_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector scalar add — IRON API design with ``@iron.jit`` compilation.

A single AIE compute core adds 1 to each element of an ``int32`` vector.
The body delegates to ``aie.iron.algorithms.transform_typed`` with an
inline lambda — the algorithm handles the ObjectFifo / Worker / Runtime
plumbing including the memtile staging tile.

Two invocation modes:

  * standalone:   ``python3 vector_scalar_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out
from aie.iron.algorithms import transform_typed
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_scalar_add(
    inp: In,
    out: Out,
    *,
    problem_size: Compile[int] = 1024,
    aie_tile_width: Compile[int] = 32,
):
    tensor_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]
    return transform_typed(lambda x: x + 1, tensor_ty, tile_size=aie_tile_width)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Scalar Add")
    add_compile_args(p, with_elf=True)
    p.add_argument(
        "--problem-size",
        type=int,
        default=1024,
        help="total elements in the input vector",
    )
    p.add_argument(
        "--aie-tile-width",
        type=int,
        default=32,
        help="elements per compute-tile sub-tile (transform_typed tile_size)",
    )
    return p


def _compile_kwargs(opts):
    return dict(
        problem_size=opts.problem_size,
        aie_tile_width=opts.aie_tile_width,
    )


def _run_and_verify(opts):
    in_t = iron.arange(1, opts.problem_size + 1, dtype=np.int32, device="npu")
    out_t = iron.zeros_like(in_t)

    vector_scalar_add(in_t, out_t, **_compile_kwargs(opts))

    expected = in_t.numpy() + 1
    actual = out_t.numpy()
    assert_pass(actual, expected, fail_msg="output does not match in + 1")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_scalar_add,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
