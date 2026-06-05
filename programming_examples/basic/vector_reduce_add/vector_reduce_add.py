# vector_reduce_add/vector_reduce_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector reduction (sum) -- ``@iron.jit`` design via the algorithms library.

A single AIE core sums an N-element int32 input vector into a 1-element int32
output.  The design body delegates to
``aie.iron.algorithms.reduce_typed``, which handles the
ObjectFifo / Worker / Runtime plumbing for the reduce shape (whole-input
single-kernel-call).

Two invocation modes (mirrors vector_vector_add):

  * standalone:   ``python3 vector_reduce_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import reduce_typed
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_reduce_add(
    a_in: In,
    c_out: Out,
    *,
    num_elements: Compile[int] = 1024,
):
    in_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]

    return reduce_typed(
        kernels.reduce_add(tile_size=num_elements, dtype=np.int32),
        in_ty,
        out_ty,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Reduce Add")
    add_compile_args(p)
    p.add_argument("-n", "--num-elements", type=int, default=1024)
    return p


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.integers(-1000, 1000, size=(opts.num_elements,), dtype=np.int32)
    in_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    out_t = iron.zeros(1, dtype=np.int32, device="npu")

    vector_reduce_add(in_t, out_t, num_elements=opts.num_elements)

    expected = np.array([in_np.sum()], dtype=np.int32)
    actual = out_t.numpy()
    assert_pass(actual, expected, fail_msg=f"expected {expected[0]}, got {actual[0]}")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_reduce_add,
        opts,
        compile_kwargs=lambda o: dict(num_elements=o.num_elements),
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
