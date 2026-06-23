# passthrough_pykernel/passthrough_pykernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Passthrough using a Python-defined kernel (pykernel) — ``@iron.jit``.

A single AIE core copies an N-element ``uint8`` vector tile by tile.  The
kernel itself is a ``@func``-decorated Python function (``passthrough_fn``)
demonstrating the pykernel pattern: write the kernel body in Python, hand it
to a ``Worker`` like any other ExternalFunction.

``vector_size`` is fixed at 4096 (matching the test.cpp / test.py defaults)
because ``@func`` decoration happens at module-import time and so its
parameter types (``line_type``) must be resolvable then.  Override via
``--vector-size`` / Makefile ``data_size=`` is therefore not supported here.

Two invocation modes:

  * standalone:   ``python3 passthrough_pykernel.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (Makefile)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import device_from_args
from aie.helpers.dialects.func import func
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

VECTOR_SIZE = 4096
LINE_SIZE = VECTOR_SIZE // 4

_LINE_TY = np.ndarray[(LINE_SIZE,), np.dtype[np.uint8]]
_VECTOR_TY = np.ndarray[(VECTOR_SIZE,), np.dtype[np.uint8]]


@func
def passthrough_fn(input: _LINE_TY, output: _LINE_TY, line_width: np.int32):
    for i in range_(line_width):
        output[i] = input[i]


@iron.jit
def passthrough_pykernel(a_in: In, b_out: Out):
    of_in = ObjectFifo(_LINE_TY, name="in")
    of_out = ObjectFifo(_LINE_TY, name="out")

    def core_fn(of_in, of_out, passthrough_fn):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        passthrough_fn(elem_in, elem_out, LINE_SIZE)
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough_fn])

    rt = Runtime()
    with rt.sequence(_VECTOR_TY, _VECTOR_TY) as (a, b):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Passthrough Pykernel")
    add_compile_args(p)
    p.add_argument(
        "-s",
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="accepted for CLI compatibility; only the build-time VECTOR_SIZE is used",
    )
    return p


def _validate(opts):
    if opts.vector_size != VECTOR_SIZE:
        sys.exit(
            f"vector_size={opts.vector_size} unsupported; this design is fixed "
            f"at {VECTOR_SIZE} (the @func pykernel parameter types are "
            f"resolved at module-import time)."
        )


def _run_and_verify(opts):
    in_t = iron.arange(1, VECTOR_SIZE + 1, dtype=np.uint8, device="npu")
    out_t = iron.zeros(VECTOR_SIZE, dtype=np.uint8, device="npu")

    passthrough_pykernel(in_t, out_t)

    expected = in_t.numpy()
    actual = out_t.numpy()
    assert_pass(actual, expected, fail_msg="output does not match input")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        passthrough_pykernel,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        device=device_from_args,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
