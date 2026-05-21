# vector_scalar_add/vector_scalar_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector scalar add — Iron API design with ``@iron.jit`` compilation.

A single AIE compute core reads ``AIE_TILE_WIDTH``-element sub-tiles, adds 1
to each element, and writes the result back.  The design preserves the
README's two-stage tiling story: ``MEM_TILE_WIDTH``-element tiles flow shim →
memtile, get split into ``AIE_TILE_WIDTH``-element tiles for the compute
core, then join back to ``MEM_TILE_WIDTH``-element tiles on the way out.

Two invocation modes (mirrors vector_scalar_mul):

  * standalone:   ``python3 vector_scalar_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, NPU2
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2()


@iron.jit
def vector_scalar_add(
    inp: In,
    out: Out,
    *,
    problem_size: Compile[int] = 1024,
    mem_tile_width: Compile[int] = 64,
    aie_tile_width: Compile[int] = 32,
):
    mem_tile_ty = np.ndarray[(mem_tile_width,), np.dtype[np.int32]]
    aie_tile_ty = np.ndarray[(aie_tile_width,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]

    of_in0 = ObjectFifo(mem_tile_ty, name="in")
    of_in1 = of_in0.cons().forward(obj_type=aie_tile_ty)

    of_out0 = ObjectFifo(aie_tile_ty, name="out")
    of_out1 = of_out0.cons().forward(obj_type=mem_tile_ty)

    def core_body(of_in1, of_out0):
        elem_in = of_in1.acquire(1)
        elem_out = of_out0.acquire(1)
        for i in range_(aie_tile_width):
            elem_out[i] = elem_in[i] + 1
        of_in1.release(1)
        of_out0.release(1)

    worker = Worker(core_body, fn_args=[of_in1.cons(), of_out0.prod()])

    rt = Runtime()
    with rt.sequence(all_data_ty, all_data_ty) as (in_tensor, out_tensor):
        rt.start(worker)
        rt.fill(of_in0.prod(), in_tensor)
        rt.drain(of_out1.cons(), out_tensor, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Scalar Add")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("--problem-size", type=int, default=1024)
    p.add_argument("--mem-tile-width", type=int, default=64)
    p.add_argument("--aie-tile-width", type=int, default=32)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument(
        "--elf-path",
        type=str,
        default=None,
        help="optional ELF-wrapped insts (for the test.cpp xrt::elf flow)",
    )
    return p


def _compile_kwargs(opts):
    return dict(
        problem_size=opts.problem_size,
        mem_tile_width=opts.mem_tile_width,
        aie_tile_width=opts.aie_tile_width,
    )


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_scalar_add.specialize(**_compile_kwargs(opts))
    spec.compile(
        xclbin_path=opts.xclbin_path,
        inst_path=opts.insts_path,
        elf_path=opts.elf_path,
    )


def _run_and_verify(opts):
    in_np = np.arange(1, opts.problem_size + 1, dtype=np.int32)
    out_np = np.zeros_like(in_np)

    in_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    out_t = iron.tensor(out_np, dtype=np.int32, device="npu")

    vector_scalar_add(in_t, out_t, **_compile_kwargs(opts))

    expected = in_np + 1
    actual = out_t.numpy()
    if not np.array_equal(actual, expected):
        sys.exit("FAIL! output does not match in + 1")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
