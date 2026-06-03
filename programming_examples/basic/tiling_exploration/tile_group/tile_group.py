# tiling_exploration/tile_group/tile_group.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Tile-group tensor access exploration — IRON + ``@iron.jit``.

Demonstrates how ``TensorTiler2D.group_tiler`` produces a single TAP
that walks the output tensor in tiled order — one ``rt.drain`` covers
all tiles in tile-major position.  The core writes values
``0, 1, 2, ...`` in element-walk order; the single TAP reorders them
into tile-major layout in the output tensor.

Three invocation modes:

  * standalone:           ``python3 tile_group.py``
  * compile-only:         ``... --xclbin-path=PATH --insts-path=PATH``
  * generate access map:  ``... --generate-access-map``  (writes
                           ``tile_group.png`` and exits)
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.util import np_dtype_to_mlir_type
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass
import aie.extras.dialects.arith as arith


@iron.jit
def tile_group(
    tensor_out: Out,
    *,
    tensor_height: Compile[int] = 8,
    tensor_width: Compile[int] = 8,
    tile_height: Compile[int] = 2,
    tile_width: Compile[int] = 2,
):
    dtype = np.int32
    tensor_size = tensor_height * tensor_width
    flattened_tensor = np.ndarray[(tensor_size,), np.dtype[dtype]]

    tap = TensorTiler2D.group_tiler(
        (tensor_height, tensor_width),
        (tile_height, tile_width),
        (tensor_height // tile_height, tensor_width // tile_width),
    )[0]

    of_out = ObjectFifo(flattened_tensor)

    def access_order(of_out):
        elemOut = of_out.acquire(1)
        for i in range_(tensor_size):
            elemOut[i] = arith.index_cast(i, to=np_dtype_to_mlir_type(dtype))
        of_out.release(1)

    worker = Worker(access_order, [of_out.prod()])

    rt = Runtime()
    with rt.sequence(flattened_tensor) as tensor_out:
        rt.start(worker)
        rt.drain(of_out.cons(), tensor_out, tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Tiling Exploration — tile group")
    add_compile_args(p)
    p.add_argument("--tensor-height", type=int, default=8)
    p.add_argument("--tensor-width", type=int, default=8)
    p.add_argument("--tile-height", type=int, default=2)
    p.add_argument("--tile-width", type=int, default=2)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="write tile_group.png and exit",
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
    out_np = np.zeros(tensor_size, dtype=dtype)
    out_t = iron.tensor(out_np, dtype=dtype, device="npu")

    tile_group(out_t, **_compile_kwargs(opts))

    expected = (
        TensorTiler2D.group_tiler(
            (opts.tensor_height, opts.tensor_width),
            (opts.tile_height, opts.tile_width),
            (
                opts.tensor_height // opts.tile_height,
                opts.tensor_width // opts.tile_width,
            ),
        )[0]
        .access_order()
        .flatten()
    )
    assert_pass(
        out_t.numpy(),
        expected,
        fail_msg="output does not match TensorTiler2D.group_tiler access order",
    )


def main():
    opts = _make_argparser().parse_args()
    if opts.generate_access_map:
        tap = TensorTiler2D.group_tiler(
            (opts.tensor_height, opts.tensor_width),
            (opts.tile_height, opts.tile_width),
            (
                opts.tensor_height // opts.tile_height,
                opts.tensor_width // opts.tile_width,
            ),
        )[0]
        tap.visualize(show_arrows=True, file_path="tile_group.png")
        return
    run_design_cli(
        tile_group,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )


if __name__ == "__main__":
    main()
