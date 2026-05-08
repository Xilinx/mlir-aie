# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""SAXPY (Z = a*X + Y) demo with a custom .cc kernel."""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, ExternalFunction, In, Out
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.utils.config import cxx_header_path


@iron.jit
def saxpy(
    input0: In, input1: In, output: Out, *, N: Compile[int], element_type: Compile[type]
):
    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    saxpy_kernel = ExternalFunction(
        "saxpy",
        source_file=os.path.join(os.path.dirname(__file__), "saxpy.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[cxx_header_path()],
    )

    def core_body(of_x, of_y, of_z, saxpy_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        saxpy_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), saxpy_kernel]
    )

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    # NOTE: saxpy.cc hardcodes the loop bound to 4096 elements.
    # data_size must match or the kernel produces silently wrong results.
    data_size = 4096
    element_type = bfloat16

    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    input1 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros_like(input0)

    saxpy(input0, input1, output, N=data_size, element_type=element_type)

    ref_vec = [3 * input0[i] + input1[i] for i in range(data_size)]
    errors = sum(1 for actual, ref in zip(output, ref_vec) if actual != ref)
    if errors:
        print(f"\nFAIL: {errors} mismatches")
        sys.exit(1)
    print("\nPASS!")


if __name__ == "__main__":
    main()
