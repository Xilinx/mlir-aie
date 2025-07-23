#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# This placed implementation uses configure_task instructions instead of
# dma_memcpy_nd in the runtime sequence configuration. It is otherwise
# identical.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import os

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_
import aie.iron as iron
from aie.iron import CoreFunction
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.extras.dialects.ext import arith
from aie.iron.device import NPU1Col1

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

m = 64
k = 64
n = 64


def ceildiv(a, b):
    return (a + b - 1) // b


@iron.jit(is_placed=False)
def gemm(a, b, c, matmul_kernel):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    trace_size = 0
    b_col_maj = 0

    print(f"M: {M}, K: {K}, N: {N}")
    # print(f"m: {m}, k: {k}, n: {n}")

    # M, K, N = 512, 512, 512  # Problem size
    # r, s, t = 8, 2, 8  # Intrinsic size int
    r, s, t = 4, 8, 4  # Intrinsic size bf16

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    dtype = bfloat16

    A_ty = np.ndarray[(M, K), np.dtype[dtype]]
    B_ty = np.ndarray[(K, N), np.dtype[dtype]]
    C_ty = np.ndarray[(M, N), np.dtype[dtype]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype]]

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

    def core_fn(of_a, of_b, of_c, matmul):

        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            matmul(0, elem_out, elem_out, elem_out)

            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(1, elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [
            fifo_A_L2L1.cons(),
            fifo_B_L2L1.cons(),
            fifo_C_L1L2.prod(),
            # zero_kernel,
            matmul_kernel,
        ],
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

    program = Program(NPU1Col1(), rt)
    return program.resolve_program(SequentialPlacer())


def do_matmul(a, b):
    # m = a.shape[0]
    # n = b.shape[1]
    M = a.shape[0]
    N = b.shape[1]
    result = iron.zeros((M, N), device="npu")

    a_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k, n), np.dtype[bfloat16]]
    c_ty = np.ndarray[(m, n), np.dtype[bfloat16]]

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    matmul_func_name = os.path.join(current_file_path, "mm.cc")

    matmul = CoreFunction(
        f"matrix_multiplication",
        source_file=matmul_func_name,
        arg_types=[np.int32, a_ty, b_ty, c_ty],
    )

    gemm(a, b, result, matmul)

    return result


def main():
    # Initialize some example tensors
    # batch_size, seq_len, hidden_size = 2, 10, 512

    # Alternative size configurations (uncomment to use):
    # batch_size, seq_len, hidden_size = 4, 20, 768  # Moderately larger
    batch_size, seq_len, hidden_size = 512, 512, 512 // 4  # Significantly larger
    # batch_size, seq_len, hidden_size = 16, 128, 2048   # Much larger
    # batch_size, seq_len, hidden_size = 32, 256, 4096   # Large (smaller Llama2)
    # batch_size, seq_len, hidden_size = 64, 512, 8192  # Very large (full Llama2)

    dtype = bfloat16

    # input_tensor = iron.randint(0, 10, (512, 512), device="npu", dtype=dtype)
    # up_weight = iron.randint(0, 10, (512, 512), device="npu", dtype=dtype)
    # output_tensor = iron.zeros(512, 512, device="npu", dtype=dtype)

    input_tensor = iron.rand((512, 512), device="npu", dtype=dtype)
    up_weight = iron.rand((512, 512), device="npu", dtype=dtype)
    output_tensor = iron.zeros((512, 512), device="npu", dtype=dtype)

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    matmul_func_name = os.path.join(current_file_path, "mm.cc")

    a_ty = np.ndarray[(m, k), np.dtype[dtype]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype]]

    matmul = CoreFunction(
        f"matrix_multiplication",
        source_file=matmul_func_name,
        arg_types=[np.int32, a_ty, b_ty, c_ty],
    )

    my_matmul(input_tensor, up_weight, output_tensor, matmul)

    print(output_tensor)

    expected = np.matmul(input_tensor.numpy(), up_weight.numpy())

    diff = np.abs(output_tensor.numpy() - expected)

    errors = np.allclose(output_tensor.numpy(), expected, atol=4)
    print("Expected: ", expected)
    print("Output: ", output_tensor.numpy())
    # print("Diff: ", diff)
    if not errors:
        print(f"Error: {errors}  out of {output_tensor.numel()} errors found")
        exit(1)

    print("Success")


if __name__ == "__main__":
    main()
