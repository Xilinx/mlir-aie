# tiling_exploration/per_tile/per_tile.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Per-tile tensor access exploration — IRON + ``@iron.jit``.

Demonstrates how ``TensorTiler2D.simple_tiler`` decomposes an output
tensor into tiles, and how one ``rt.drain`` per tile reorders the
otherwise-sequential per-element write order into a tiled layout.  The
core produces values ``0, 1, 2, ...`` in element-walk order; the drain
TAPs scatter them into tile-major position in the output tensor.

Three invocation modes:

  * standalone:           ``python3 per_tile.py``
  * compile-only:         ``... --xclbin-path=PATH --insts-path=PATH``
  * generate access map:  ``... --generate-access-map``  (writes
                           ``per_tile.png`` and exits)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, Compile, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def per_tile(
    tensor_out: Out,
    *,
    tensor_height: Compile[int] = 8,
    tensor_width: Compile[int] = 8,
    tile_height: Compile[int] = 2,
    tile_width: Compile[int] = 2,
):
    dtype = np.int32
    tensor_size = tensor_height * tensor_width
    tile_size = tile_height * tile_width
    flattened_tensor = np.ndarray[(tensor_size,), np.dtype[dtype]]
    flattened_tile = np.ndarray[(tile_size,), np.dtype[dtype]]

    tiler = TensorTiler2D.simple_tiler(
        (tensor_height, tensor_width), (tile_height, tile_width)
    )

    of_out = ObjectFifo(flattened_tile)
    access_counter = Buffer(initial_value=np.array([0], dtype=dtype))

    def access_order(of_out, counter_buf):
        elemOut = of_out.acquire(1)
        for i in range_(tile_size):
            elemOut[i] = counter_buf[0]
            counter_buf[0] += 1
        of_out.release(1)

    worker = Worker(access_order, [of_out.prod(), access_counter])

    rt = Runtime()
    with rt.sequence(flattened_tensor) as tensor_out:
        rt.start(worker)
        for t in tiler:
            rt.drain(of_out.cons(), tensor_out, t, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Tiling Exploration — per-tile")
    add_compile_args(p)
    p.add_argument("--tensor-height", type=int, default=8)
    p.add_argument("--tensor-width", type=int, default=8)
    p.add_argument("--tile-height", type=int, default=2)
    p.add_argument("--tile-width", type=int, default=2)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="write per_tile.png and exit",
    )
    return p


def _compile_kwargs(opts):
    return dict(
        tensor_height=opts.tensor_height,
        tensor_width=opts.tensor_width,
        tile_height=opts.tile_height,
        tile_width=opts.tile_width,
    )


def _run_and_verify(opts):
    dtype = np.int32
    tensor_size = opts.tensor_height * opts.tensor_width
    out_t = iron.zeros(tensor_size, dtype=dtype, device="npu")

    per_tile(out_t, **_compile_kwargs(opts))

    expected = (
        TensorTiler2D.simple_tiler(
            (opts.tensor_height, opts.tensor_width),
            (opts.tile_height, opts.tile_width),
        )
        .access_order()
        .flatten()
    )
    assert_pass(
        out_t.numpy(),
        expected,
        fail_msg="output does not match TensorTiler2D.simple_tiler access order",
    )


def main():
    opts = _make_argparser().parse_args()
    if opts.generate_access_map:
        tiler = TensorTiler2D.simple_tiler(
            (opts.tensor_height, opts.tensor_width),
            (opts.tile_height, opts.tile_width),
        )
        tiler.visualize(file_path="per_tile.png")
        return
    run_design_cli(
        per_tile,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
