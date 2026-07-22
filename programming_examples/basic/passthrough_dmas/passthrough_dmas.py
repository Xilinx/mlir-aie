# passthrough_dmas/passthrough_dmas.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Passthrough DMAs — IRON API design with ``@iron.jit`` compilation.

No compute tile: data flows shim → memtile → shim via
``ObjectFifo.forward()``, exercising the implicit-copy DMA path.

Two invocation modes:

  * standalone:   ``python3 passthrough_dmas.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``       (NPU)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import AnyShimTile
from aie.utils.hostruntime.argparse import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

LINE_SIZE = 1024  # transfer chunk; N must be a multiple of this


@iron.jit
def passthrough_dmas(
    a_in: In,
    _b_unused: In,
    c_out: Out,
    *,
    n: CompileTime[int] = 4096,
):
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]
    line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    def sequence(a, _, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [vector_ty, vector_ty, vector_ty],
        fn_args=[of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
    )

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Passthrough DMAs")
    add_compile_args(
        p,
        dev_choices=("npu", "npu2"),
        with_emit_mlir=True,
        with_elf=True,
    )
    p.add_argument("-n", "--length", type=int, default=4096, help="elements")
    return p


def _validate(opts):
    if opts.length % LINE_SIZE != 0:
        sys.exit(f"--length ({opts.length}) must be a multiple of {LINE_SIZE}")


def _compile_kwargs(opts):
    return dict(n=opts.length)


def _run_and_verify(opts):
    a_t = iron.arange(1, opts.length + 1, dtype=np.int32, device="npu")
    b_t = iron.zeros_like(a_t)  # unused 2nd buffer
    c_t = iron.zeros_like(a_t)

    passthrough_dmas(a_t, b_t, c_t, **_compile_kwargs(opts))

    assert_pass(c_t.numpy(), a_t.numpy(), fail_msg="output does not match input")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        passthrough_dmas,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
