# bfp_conversion.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""bf16 → bfp16ebs8 conversion + bfp16 matmul — ``@iron.jit`` IRON design.

Two AIE2P cores form a pipeline: the first converts two bf16 input tiles
to bfp16ebs8, the second multiplies the converted tiles.  Strix-only
(uses ``v8bfp16ebs8``).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_dtypes import bfloat16
from aie.dialects.aiex import v8bfp16ebs8

import aie.iron as iron
from aie.iron import ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

N_IN = 64
N_OUT = 8

_TENSOR_BF16_TY = np.ndarray[(N_IN,), np.dtype[bfloat16]]
_TILE_BF16_TY = np.ndarray[(N_IN,), np.dtype[bfloat16]]
_TENSOR_BFP16_TY = np.ndarray[(N_OUT,), np.dtype[v8bfp16ebs8]]
_TILE_BFP16_TY = np.ndarray[(N_OUT,), np.dtype[v8bfp16ebs8]]

_KERNEL_SRC = Path(__file__).resolve().parent / "kernel.cc"


@iron.jit
def bfp_conversion(a_in: In, b_in: In, c_out: Out):
    conversion_kernel = ExternalFunction(
        "bf16_to_bfp_conversion",
        source_file=str(_KERNEL_SRC),
        arg_types=[_TILE_BF16_TY, _TILE_BF16_TY, _TILE_BFP16_TY, _TILE_BFP16_TY],
    )
    multiplication_kernel = ExternalFunction(
        "bfp16_matrix_multiplication",
        source_file=str(_KERNEL_SRC),
        arg_types=[_TILE_BFP16_TY, _TILE_BFP16_TY, _TILE_BFP16_TY],
    )

    of_in1 = ObjectFifo(_TILE_BF16_TY, name="in1")
    of_in2 = ObjectFifo(_TILE_BF16_TY, name="in2")
    of_intermediate1 = ObjectFifo(_TILE_BFP16_TY, name="intermediate1")
    of_intermediate2 = ObjectFifo(_TILE_BFP16_TY, name="intermediate2")
    of_out = ObjectFifo(_TILE_BFP16_TY, name="out")

    def conversion_core(of_in1, of_in2, of_inter1, of_inter2, kernel):
        for _ in range_(sys.maxsize):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out1 = of_inter1.acquire(1)
            elem_out2 = of_inter2.acquire(1)
            kernel(elem_in1, elem_in2, elem_out1, elem_out2)
            of_in1.release(1)
            of_in2.release(1)
            of_inter1.release(1)
            of_inter2.release(1)

    def multiplication_core(of_inter1, of_inter2, of_out, kernel):
        for _ in range_(sys.maxsize):
            elem_in1 = of_inter1.acquire(1)
            elem_in2 = of_inter2.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in1, elem_in2, elem_out)
            of_inter1.release(1)
            of_inter2.release(1)
            of_out.release(1)

    workers = [
        Worker(
            conversion_core,
            fn_args=[
                of_in1.cons(),
                of_in2.cons(),
                of_intermediate1.prod(),
                of_intermediate2.prod(),
                conversion_kernel,
            ],
        ),
        Worker(
            multiplication_core,
            fn_args=[
                of_intermediate1.cons(),
                of_intermediate2.cons(),
                of_out.prod(),
                multiplication_kernel,
            ],
        ),
    ]

    rt = Runtime()
    with rt.sequence(_TENSOR_BF16_TY, _TENSOR_BF16_TY, _TENSOR_BFP16_TY) as (A, B, C):
        rt.start(*workers)
        rt.fill(of_in1.prod(), A)
        # Aligning dot products with bfp blocks requires transposing the second
        # matrix before conversion to bfp; bf16's element size (2B) precludes a
        # 4B-aligned transpose at this level, so transposition happens inside
        # the multiplication kernel.
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE bf16 -> bfp16 Conversion + GEMM")
    add_compile_args(p, default_dev="npu2")
    return p


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        bfp_conversion,
        opts,
        compile_kwargs={},
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
