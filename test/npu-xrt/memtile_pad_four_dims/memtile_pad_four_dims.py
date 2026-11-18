# memtile_pad_four_dims/memtile_pad_four_dims.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %run_on_npu2% %python %s -d npu2 --dim 2 --before 1 --after 0
# RUN: %run_on_npu2% %python %s -d npu2 --dim 2 --before 0 --after 1
# RUN: %run_on_npu2% %python %s -d npu2 --dim 2 --before 1 --after 1
# RUN: %run_on_npu2% %python %s -d npu2 --dim 2 --before 0 --after 8
# RUN: %run_on_npu2% %python %s -d npu2 --dim 1 --before 1 --after 1
# RUN: %run_on_npu2% %python %s -d npu2 --dim 0 --before 0 --after 8
# RUN: %run_on_npu2% %python %s -d npu2 --walk gemm-a --dim 2 --before 0 --after 8
# RUN: %run_on_npu2% %python %s -d npu2 --walk gemm-b --dim 2 --before 0 --after 8
"""Padding in a four-dimensional MemTile walk.

A shim tile sends a buffer to a MemTile, which sends it back through an
MM2S walk over four dimensions, d3 to d0, with zero padding on one of the
inner three (d3 has no padding fields). The result is checked against
``np.pad`` of the same walk.

Walks (sizes and strides in elements, d3 first):

- ``cube``: [2, 2, 2, 8] over a 64-element buffer.
- ``gemm-a`` and ``gemm-b``: the A and B walks of a bf16 GEMM whose last
  K step is padded from 192 to 256; int16 carries the bf16 bits.
"""

import argparse
import math

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

# name: (buffer elements, dtype, sizes, strides)
WALKS = {
    "cube": (64, np.int32, [2, 2, 2, 8], [32, 16, 8, 1]),
    "gemm-a": (32 * 936, np.int16, [4, 24, 8, 8], [7680, 8, 960, 1]),
    "gemm-b": (192 * 32, np.int16, [4, 24, 8, 8], [8, 256, 32, 1]),
}


def _pads(dim, before, after):
    pads = [(0, 0)] * 4
    pads[3 - dim] = (before, after)
    return pads


def _expected(src, sizes, strides, pads):
    item = src.dtype.itemsize
    walk = np.lib.stride_tricks.as_strided(src, sizes, [s * item for s in strides])
    return np.pad(walk, pads).reshape(-1)


@iron.jit
def memtile_pad(
    a_in: In,
    c_out: Out,
    *,
    walk: CompileTime[str] = "cube",
    dim: CompileTime[int] = 2,
    before: CompileTime[int] = 0,
    after: CompileTime[int] = 1,
):
    n, dtype, sizes, strides = WALKS[walk]
    pads = _pads(dim, before, after)
    tap = TensorAccessPattern((n,), 0, sizes, strides).pad(pads)
    n_out = math.prod(s + b + a for s, (b, a) in zip(sizes, pads))
    in_ty = np.ndarray[(n,), np.dtype[dtype]]
    out_ty = np.ndarray[(n_out,), np.dtype[dtype]]

    src = ObjectFifo(in_ty, name="src")
    dst = src.cons().forward(obj_type=out_ty, to_stream=tap, name="dst")

    def sequence(a, c, src_prod, dst_cons):
        src_prod.fill(a)
        dst_cons.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, out_ty, src.prod(tile=AnyShimTile), dst.cons(tile=AnyShimTile)],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    return dict(walk=opts.walk, dim=opts.dim, before=opts.before, after=opts.after)


def _run_and_verify(opts):
    n, dtype, sizes, strides = WALKS[opts.walk]
    src = np.arange(1, n + 1).astype(dtype)
    expected = _expected(src, sizes, strides, _pads(opts.dim, opts.before, opts.after))
    a_t = iron.tensor(src, device="npu")
    c_t = iron.zeros(expected.size, dtype=dtype, device="npu")

    memtile_pad(a_t, c_t, **_compile_kwargs(opts))

    assert_pass(
        c_t.numpy(),
        expected,
        fail_msg=f"{opts.walk}: d{opts.dim} padding ({opts.before}, {opts.after})",
    )


def main():
    p = argparse.ArgumentParser(prog="MemTile four-dimensional padding")
    add_compile_args(p, dev_choices=("npu2",), default_dev="npu2", with_emit_mlir=True)
    p.add_argument("--walk", choices=sorted(WALKS), default="cube")
    p.add_argument(
        "--dim", type=int, choices=[0, 1, 2], default=2, help="hardware dim to pad"
    )
    p.add_argument("--before", type=int, default=0)
    p.add_argument("--after", type=int, default=1)
    opts = p.parse_args()
    run_design_cli(
        memtile_pad,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
