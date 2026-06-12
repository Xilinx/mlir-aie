# single_core_designs/vector_reduce_max.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-core vector reduction (max) -- ``@iron.jit`` design via the algorithms library.

A single AIE core finds the maximum of an N-element input vector, producing a
1-element output.  Supports ``int32`` and ``bfloat16`` element types (binding
the corresponding ``reduce_max_vector`` / ``reduce_max_vector_bfloat16``
kernel symbol from ``reduce_max.cc``).

The design body delegates to ``aie.iron.algorithms.reduce_typed``; trace is
threaded through via the library's ``trace_size`` kwarg.

Two invocation modes:

  * standalone:   ``python3 vector_reduce_max.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels, str_to_dtype
from aie.iron.algorithms import reduce_typed
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_reduce_max(
    a_in: In,
    c_out: Out,
    *,
    num_elements: CompileTime[int] = 2048,
    dtype: CompileTime[type] = np.int32,
    trace_size: CompileTime[int] = 0,
):
    # Output buffer is always 4 bytes (the size of one int32) -- the original
    # test.cpp / test.py treat it as a 4-byte slot regardless of element dtype,
    # so bf16 fills 2 elements (the second is garbage) while int32 fills 1.
    out_num_elements = 4 // np.dtype(dtype).itemsize
    in_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_num_elements,), np.dtype[dtype]]

    return reduce_typed(
        kernels.reduce_max(tile_size=num_elements, dtype=dtype),
        in_ty,
        out_ty,
        trace_size=trace_size,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Single-Core Vector Reduce Max")
    add_compile_args(p)
    p.add_argument("-i1s", "--in1_size", type=int, default=8192, help="bytes")
    p.add_argument("-os", "--out_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument("-dt", "--dtype", type=str, default="i32", choices=["i32", "bf16"])
    add_trace_arg(p)
    return p


def _validate(opts):
    if opts.in1_size % 64 != 0 or opts.in1_size < 512:
        sys.exit("in1_size must be a multiple of 64 and >= 512")
    if opts.out_size != 4:
        sys.exit("out_size must be 4 (1 x 4-byte scalar)")


def _compile_kwargs(opts):
    dtype = str_to_dtype(opts.dtype)
    num_elements = opts.in1_size // np.dtype(dtype).itemsize
    return dict(
        num_elements=num_elements,
        dtype=dtype,
        trace_size=opts.trace_size,
    )


def _run_and_verify(opts):
    dtype = str_to_dtype(opts.dtype)
    num_elements = opts.in1_size // np.dtype(dtype).itemsize

    out_num_elements = 4 // np.dtype(dtype).itemsize
    rng = np.random.default_rng(0)
    if opts.dtype == "i32":
        in_np = rng.integers(-1000, 1000, size=(num_elements,), dtype=np.int32)
    else:  # bf16
        in_np = rng.uniform(-1000.0, 1000.0, size=(num_elements,)).astype(dtype)
    in_t = iron.tensor(in_np, dtype=dtype, device="npu")
    out_t = iron.zeros(out_num_elements, dtype=dtype, device="npu")

    vector_reduce_max(in_t, out_t, **_compile_kwargs(opts))

    expected_max = in_np.max()
    actual_max = out_t.numpy()[0]  # the first slot holds the reduction result
    assert_pass(
        actual_max, expected_max, fail_msg=f"expected {expected_max}, got {actual_max}"
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_reduce_max,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
