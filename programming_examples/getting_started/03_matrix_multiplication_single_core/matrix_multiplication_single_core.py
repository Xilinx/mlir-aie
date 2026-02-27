# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys
import os

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.config import cxx_header_path


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit(is_placed=False)
def matrix_multiplication_single_core(input0, input1, output):
    # Problem size
    # - matrix0 shapes: (M, K)
    # - matrix1 shapes: (K, N)
    M, K, N = input0.shape[0], input0.shape[1], input1.shape[1]
    m, k, n = 64, 64, 64  # Tile size moved to/from the compute cores via mem tiles
    r, s, t = 8, 2, 8  # AIE kernel intrinsic size

    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    A_ty = np.ndarray[(M, K), np.dtype[element_type]]
    B_ty = np.ndarray[(K, N), np.dtype[element_type]]
    C_ty = np.ndarray[(M, N), np.dtype[element_type]]
    a_ty = np.ndarray[(m, k), np.dtype[element_type]]
    b_ty = np.ndarray[(k, n), np.dtype[element_type]]
    c_ty = np.ndarray[(m, n), np.dtype[element_type]]

    # The following ObjectFIFOs route m*k-, k*n-, and m*n-sized subtiles
    # (objects) to/from the compute cores via mem tiles, rearranging their data
    # into r*s-, s*t-, and r*t-sized sub-subtiles.

    fifo_A_L3L2 = ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=tap_C_L1L2.transformation_dims, name="C_L2L3"
    )

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel repeatedly acquires one subtile of A and B, multiplies them,
    # and accumulates the result on top of C. As these tiles come in, the DMAs
    # will have rearranged them into r*s-, s*t-, and r*t-sized subtiles, which
    # the computation kernel relies on.

    matmul_kernel = ExternalFunction(
        "matrix_multiplication",
        source_file=os.path.join(os.path.dirname(__file__), "matrix_multiplication.cc"),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=[cxx_header_path()],
    )

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            for i in range_(m):
                for j in range_(n):
                    elem_out[i, j] = 0
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), matmul_kernel],
    )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    # The data movement patterns from DRAM divide the input matrices (sizes
    # M*K, K*N) into m*k- and k*n-sized subtiles and produce output into C in
    # m*n-sized subtiles. Each single "task group" encompasses all data
    # movement required for a single row of the output matrix.

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        for tile_row in range(M // m):
            task_group = rt.task_group()
            rt.fill(fifo_A_L3L2.prod(), A, tap=a_taps[tile_row], task_group=task_group)
            rt.fill(fifo_B_L3L2.prod(), B, tap=b_tap, task_group=task_group)
            rt.drain(
                fifo_C_L2L3.cons(),
                C,
                tap=c_taps[tile_row],
                task_group=task_group,
                wait=True,
            )
            rt.finish_task_group(task_group)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program()


def main():
    # Define tensor shapes and data types
    M, K, N = 512, 512, 512
    element_type = np.int16

    # Construct an input tensors and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.randint(0, 256, (M, K), dtype=element_type, device="npu")
    input1 = iron.randint(0, 256, (K, N), dtype=element_type, device="npu")
    output = iron.zeros(M * N, dtype=element_type, device="npu")

    # Generate reference pattern
    ref_vec = np.matmul(input0.numpy(), input1.numpy())

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    matrix_multiplication_single_core(input0, input1, output)

    # Check the correctness of the result
    e = np.equal(ref_vec.flatten(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # If the result is correct, exit with a success code
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
