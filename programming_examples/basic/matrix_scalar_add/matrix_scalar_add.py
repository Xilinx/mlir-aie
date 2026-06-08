# matrix_scalar_add/matrix_scalar_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Matrix scalar add — IRON API design with ``@iron.jit`` compilation.

A single AIE compute core reads one ``TILE_HEIGHT x TILE_WIDTH`` tile from
the top-left corner of a ``MATRIX_HEIGHT x MATRIX_WIDTH`` matrix (via
``TensorTiler2D.simple_tiler``), adds 1 to each element of that tile, and
writes it back, leaving the rest of the matrix unchanged — a
subtile-region DMA access pattern.  Default config: 16x128 matrix,
8x16 tile.

Three invocation modes:

  * standalone:   ``python3 matrix_scalar_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
  * emit-MLIR:    ``... -d xcvc1902 --emit-mlir``               (vck5000)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import device_from_args
from aie.helpers.taplib import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def matrix_scalar_add(
    inp: In,
    out: Out,
    *,
    matrix_height: Compile[int] = 16,
    matrix_width: Compile[int] = 128,
    tile_height: Compile[int] = 8,
    tile_width: Compile[int] = 16,
):
    matrix_shape = (matrix_height, matrix_width)
    tile_shape = (tile_height, tile_width)

    matrix_ty = np.ndarray[matrix_shape, np.dtype[np.int32]]
    tile_ty = np.ndarray[tile_shape, np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in0")
    of_out = ObjectFifo(tile_ty, name="out0")

    def core_fn(of_in1, of_out1):
        elem_in = of_in1.acquire(1)
        elem_out = of_out1.acquire(1)
        for i in range_(tile_height):
            for j in range_(tile_width):
                elem_out[i, j] = elem_in[i, j] + 1
        of_in1.release(1)
        of_out1.release(1)

    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod()])

    tap = TensorTiler2D.simple_tiler(matrix_shape, tile_shape)[0]

    rt = Runtime()
    with rt.sequence(matrix_ty, matrix_ty) as (in_tensor, out_tensor):
        rt.start(worker)
        rt.fill(of_in.prod(), in_tensor, tap)
        rt.drain(of_out.cons(), out_tensor, tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Scalar Add")
    add_compile_args(p, dev_choices=("npu", "npu2", "xcvc1902"), with_emit_mlir=True)
    p.add_argument("--matrix-height", type=int, default=16)
    p.add_argument("--matrix-width", type=int, default=128)
    p.add_argument("--tile-height", type=int, default=8)
    p.add_argument("--tile-width", type=int, default=16)
    return p


def _compile_kwargs(opts):
    return dict(
        matrix_height=opts.matrix_height,
        matrix_width=opts.matrix_width,
        tile_height=opts.tile_height,
        tile_width=opts.tile_width,
    )


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    in_np = rng.integers(
        -1000, 1000, size=(opts.matrix_height, opts.matrix_width), dtype=np.int32
    )
    in_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    out_t = iron.zeros_like(in_t)

    matrix_scalar_add(in_t, out_t, **_compile_kwargs(opts))

    expected = in_np.copy()
    expected[: opts.tile_height, : opts.tile_width] += 1
    actual = out_t.numpy().reshape(opts.matrix_height, opts.matrix_width)
    assert_pass(
        actual,
        expected,
        fail_msg="output does not match in + 1 on the top-left tile (subtile DMA)",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        matrix_scalar_add,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
