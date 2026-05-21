# passthrough_dmas/passthrough_dmas.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough DMAs — Iron API design with ``@iron.jit`` compilation.

No compute tile: data flows shim → memtile → shim via an ObjectFifo
``forward()``, exercising the implicit-copy DMA path.

Three invocation modes (mirrors matrix_scalar_add):

  * standalone:   ``python3 passthrough_dmas.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``       (NPU)
  * emit-MLIR:    ``... -d xcvc1902 --emit-mlir``                    (vck5000)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2, XCVC1902
from aie.utils.hostruntime import set_current_device

LINE_SIZE = 1024  # transfer chunk; N must be a multiple of this


def _device_for(dev_str):
    if dev_str == "npu":
        return NPU1Col1()
    if dev_str == "npu2":
        return NPU2()
    if dev_str == "xcvc1902":
        return XCVC1902()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


@iron.jit
def passthrough_dmas(
    a_in: In,
    _b_unused: In,
    c_out: Out,
    *,
    n: Compile[int] = 4096,
):
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    rt = Runtime()
    with rt.sequence(vector_ty, vector_ty, vector_ty) as (a, _, c):
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Passthrough DMAs")
    p.add_argument(
        "-d", "--dev", type=str, choices=["npu", "npu2", "xcvc1902"], default="npu"
    )
    p.add_argument("-n", "--length", type=int, default=4096, help="elements")
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the resolved MLIR module to stdout (legacy aiecc / vck5000 path)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument(
        "--elf-path",
        type=str,
        default=None,
        help="optional ELF-wrapped insts (for the test.cpp xrt::elf flow)",
    )
    return p


def _validate(opts):
    if opts.length % LINE_SIZE != 0:
        sys.exit(f"--length ({opts.length}) must be a multiple of {LINE_SIZE}")


def _compile_kwargs(opts):
    return dict(n=opts.length)


def _emit_mlir(opts):
    set_current_device(_device_for(opts.dev))
    print(passthrough_dmas.as_mlir(None, None, None, **_compile_kwargs(opts)))


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = passthrough_dmas.specialize(**_compile_kwargs(opts))
    spec.compile(
        xclbin_path=opts.xclbin_path,
        inst_path=opts.insts_path,
        elf_path=opts.elf_path,
    )


def _run_and_verify(opts):
    in_np = np.arange(1, opts.length + 1, dtype=np.int32)
    b_np = np.zeros_like(in_np)  # unused 2nd buffer
    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    b_t = iron.tensor(b_np, dtype=np.int32, device="npu")
    c_t = iron.tensor(out_np, dtype=np.int32, device="npu")

    passthrough_dmas(a_t, b_t, c_t, **_compile_kwargs(opts))

    if not np.array_equal(c_t.numpy(), in_np):
        sys.exit("FAIL! output does not match input")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.emit_mlir:
        _emit_mlir(opts)
        return
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
