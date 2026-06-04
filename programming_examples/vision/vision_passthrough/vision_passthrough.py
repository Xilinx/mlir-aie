# vision/vision_passthrough/vision_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vision passthrough -- ``@iron.jit`` line-based image copy.

A single AIE core copies a ``width x height`` 8-bit image one line at a
time via ``passThroughLine`` (``-DBIT_WIDTH=8`` from
``aie_kernels/generic/passThrough.cc``).
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


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
    add_compile_args(p)
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    return p


def _compile_kwargs(opts):
    return dict(width=opts.width, height=opts.height)


def _run_and_verify(opts):
    tensor_size = opts.width * opts.height
    rng = np.random.default_rng(0)
    in_np = rng.integers(-128, 127, size=(tensor_size,), dtype=np.int8)
    zeros_np = np.zeros((tensor_size,), dtype=np.int8)

    in_t = iron.tensor(in_np, dtype=np.int8, device="npu")
    out_t = iron.tensor(zeros_np, dtype=np.int8, device="npu")
    third_t = iron.tensor(zeros_np, dtype=np.int8, device="npu")

    vision_passthrough(in_t, third_t, out_t, **_compile_kwargs(opts))

    assert_pass(out_t.numpy(), in_np, fail_msg="output does not match input")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vision_passthrough,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
