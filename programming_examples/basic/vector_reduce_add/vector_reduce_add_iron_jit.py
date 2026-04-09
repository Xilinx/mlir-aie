# vector_reduce_add/vector_reduce_add_iron_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


@iron.jit
def my_reduce_add(
    input_tensor: In,
    output_tensor: Out,
    *,
    N: Compile[int] = 1024,
):
    in_ty = np.ndarray[(N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]

    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    reduce_add_fn = iron.kernels.reduce_add(tile_size=N)

    def core_body(of_in, of_out, reduce_add_vector):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        reduce_add_vector(elem_in, elem_out, N)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), reduce_add_fn])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    N = 1024
    input_tensor = iron.randint(0, 100, (N,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1,), dtype=np.int32, device="npu")

    my_reduce_add(input_tensor, output_tensor, N=N)

    expected = int(np.sum(input_tensor.numpy()))
    computed = int(output_tensor.numpy()[0])

    if expected == computed:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print(f"\nFAIL! Expected {expected} but got {computed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
