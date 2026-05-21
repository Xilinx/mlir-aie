# vision/vision_passthrough/vision_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vision passthrough -- ``@iron.jit`` line-based image copy.

A single AIE core copies a ``width x height`` 8-bit image one line at a
time.  The ``passThroughLine`` kernel (selected via ``-DBIT_WIDTH=8`` from
``aie_kernels/generic/passThrough.cc``) does the per-line memcpy.

``aiecc_flags=["--alloc-scheme=basic-sequential"]`` matches the pre-merge
Makefile's aiecc invocation; this baseline allocator produces the existing
single-bank buffer layout (vision pipelines do not benefit from the 4-bank
distribution that the default allocator gives matmul-style designs).

Two invocation modes:

  * standalone:   ``python3 vision_passthrough.py``
        (verifies in-Python that output bytes equal input bytes)
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2Col1()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def vision_passthrough(
    in_tensor: In,
    _unused: In,
    out_tensor: Out,
    *,
    width: Compile[int] = 1920,
    height: Compile[int] = 1080,
):
    tensor_size = width * height
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int8]]
    line_ty = np.ndarray[(width,), np.dtype[np.uint8]]

    pass_through_line = kernels.passthrough(tile_size=width, dtype=np.uint8)

    of_in = ObjectFifo(line_ty, name="in")
    of_out = ObjectFifo(line_ty, name="out")

    def passthrough_fn(of_in, of_out, kernel):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        kernel(elem_in, elem_out, width)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(passthrough_fn, [of_in.cons(), of_out.prod(), pass_through_line])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a, _, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vision Passthrough")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vision_passthrough.specialize(width=opts.width, height=opts.height)
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    tensor_size = opts.width * opts.height
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    zeros_np = np.zeros((tensor_size,), dtype=np.int8)

    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    out_t = iron.tensor(zeros_np, dtype=np.int8, device="npu")
    third_t = iron.tensor(zeros_np, dtype=np.int8, device="npu")

    vision_passthrough(in_t, third_t, out_t, width=opts.width, height=opts.height)

    actual = out_t.numpy()
    if not np.array_equal(actual, in_np):
        sys.exit("FAIL! output does not match input")

    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
