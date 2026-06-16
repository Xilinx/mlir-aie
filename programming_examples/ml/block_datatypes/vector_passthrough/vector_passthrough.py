# vector_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""bfp16ebs8 vector passthrough — ``@iron.jit`` IRON design.

A single AIE2P core copies a 32-element ``v8bfp16ebs8`` tensor through
a 16-element tile.  Strix-only because ``v8bfp16ebs8`` is an AIE2P
intrinsic type.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from aie.dialects.aiex import v8bfp16ebs8

import aie.iron as iron
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

N = 32
n = 16

_TENSOR_TY = np.ndarray[(N,), np.dtype[v8bfp16ebs8]]
_TILE_TY = np.ndarray[(n,), np.dtype[v8bfp16ebs8]]

_KERNEL_SRC = Path(__file__).resolve().parent / "kernel.cc"


@iron.jit
def vector_passthrough(a_in: In, b_out: Out):
    passthrough_func = ExternalFunction(
        "bfp16_passthrough_vectorized",
        source_file=str(_KERNEL_SRC),
        arg_types=[_TILE_TY, _TILE_TY],
    )

    of_in = ObjectFifo(_TILE_TY, name="in")
    of_out = ObjectFifo(_TILE_TY, name="out")

    def core_fn(of_in, of_out, kernel):
        for _ in range_(sys.maxsize):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough_func])

    rt = Runtime()

    def sequence(A, B):
        of_in.prod().fill(A)
        of_out.cons().drain(B, wait=True)

    rt.sequence(sequence, [_TENSOR_TY, _TENSOR_TY])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE bfp16 Vector Passthrough")
    add_compile_args(p, default_dev="npu2")
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_passthrough,
        opts,
        compile_kwargs={},
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
