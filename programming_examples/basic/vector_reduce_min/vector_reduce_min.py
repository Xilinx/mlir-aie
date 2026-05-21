# vector_reduce_min/vector_reduce_min.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector reduction (min) -- ``@iron.jit`` design via the algorithms library.

A single AIE core finds the minimum of an N-element int32 input vector,
producing a 1-element int32 output.  The design body delegates to
``aie.iron.algorithms.reduce_typed``.

Two invocation modes:

  * standalone:   ``python3 vector_reduce_min.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import reduce_typed
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2Col1()


@iron.jit
def vector_reduce_min(
    a_in: In,
    c_out: Out,
    *,
    num_elements: Compile[int] = 1024,
):
    in_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]

    return reduce_typed(
        kernels.reduce_min(tile_size=num_elements, dtype=np.int32),
        in_ty,
        out_ty,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Reduce Min")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-n", "--num-elements", type=int, default=1024)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_reduce_min.specialize(num_elements=opts.num_elements)
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.integers(-1000, 1000, size=(opts.num_elements,), dtype=np.int32)
    out_np = np.zeros((1,), dtype=np.int32)

    in_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    out_t = iron.tensor(out_np, dtype=np.int32, device="npu")

    vector_reduce_min(in_t, out_t, num_elements=opts.num_elements)

    expected = np.array([in_np.min()], dtype=np.int32)
    actual = out_t.numpy()
    if not np.array_equal(actual, expected):
        sys.exit(f"FAIL! expected {expected[0]}, got {actual[0]}")

    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
