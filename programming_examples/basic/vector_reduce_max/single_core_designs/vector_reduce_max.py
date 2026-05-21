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
import os
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, ExternalFunction, In, Out, str_to_dtype
from aie.iron.algorithms import reduce_typed
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils.hostruntime import set_current_device

_KERNELS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "aie_kernels", "aie2"
    )
)


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2Col1()


@iron.jit
def vector_reduce_max(
    a_in: In,
    c_out: Out,
    *,
    num_elements: Compile[int] = 2048,
    dtype: Compile[type] = np.int32,
    trace_size: Compile[int] = 0,
):
    # Output buffer is always 4 bytes (the size of one int32) -- the original
    # test.cpp / test.py treat it as a 4-byte slot regardless of element dtype,
    # so bf16 fills 2 elements (the second is garbage) while int32 fills 1.
    out_num_elements = 4 // np.dtype(dtype).itemsize
    in_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_num_elements,), np.dtype[dtype]]

    # reduce_max.cc exports two symbols: reduce_max_vector (int32) and
    # reduce_max_vector_bfloat16.  Pick the one matching the element type.
    if (
        dtype == np.dtype("bfloat16").type
        or getattr(dtype, "__name__", "") == "bfloat16"
    ):
        symbol = "reduce_max_vector_bfloat16"
    else:
        symbol = "reduce_max_vector"

    reduce_max = ExternalFunction(
        symbol,
        source_file=os.path.join(_KERNELS_DIR, "reduce_max.cc"),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=[_KERNELS_DIR],
    )

    return reduce_typed(reduce_max, in_ty, out_ty, trace_size=trace_size)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Single-Core Vector Reduce Max")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-i1s", "--in1_size", type=int, default=8192, help="bytes")
    p.add_argument("-os", "--out_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument("-dt", "--dtype", type=str, default="i32", choices=["i32", "bf16"])
    p.add_argument("-t", "--trace_size", type=int, default=0)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
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


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_reduce_max.specialize(**_compile_kwargs(opts))
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    dtype = str_to_dtype(opts.dtype)
    num_elements = opts.in1_size // np.dtype(dtype).itemsize

    out_num_elements = 4 // np.dtype(dtype).itemsize
    rng = np.random.default_rng(0)
    if opts.dtype == "i32":
        in_np = rng.integers(-1000, 1000, size=(num_elements,), dtype=np.int32)
    else:  # bf16
        in_np = rng.uniform(-1000.0, 1000.0, size=(num_elements,)).astype(dtype)
    out_np = np.zeros((out_num_elements,), dtype=dtype)

    in_t = iron.tensor(in_np, dtype=dtype, device="npu")
    out_t = iron.tensor(out_np, dtype=dtype, device="npu")

    vector_reduce_max(in_t, out_t, **_compile_kwargs(opts))

    expected_max = in_np.max()
    actual_max = out_t.numpy()[0]  # the first slot holds the reduction result
    if actual_max != expected_max:
        sys.exit(f"FAIL! expected {expected_max}, got {actual_max}")

    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
