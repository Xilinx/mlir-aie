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
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_
from aie.iron.dtype import str_to_dtype


microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        maybe_taps = my_matmul(
            args.dev,
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.emulate_bf16_mmul_with_bfp16,
            args.trace_size,
            args.generate_taps,
        )
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    generate_taps=False,
):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Matrix B: KxN, submatrices b: kxn
    k_x_N = k * N

    # Output Matrix C: MxN
    m_x_N = m * N

    C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    if dev == "npu":
        dev_ty = AIEDevice.npu1_1col
    else:
        dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # AIE Core Function declarations
        func_type = "" if vectorized else "scalar_"
        zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[c_ty])
        matmul_func_name = f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}"
        matmul = external_func(
            matmul_func_name,
            inputs=[a_ty, b_ty, c_ty],
        )

        # Tile declarations
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2_col, compute_tile2_row = 0, 2
        compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", shim_tile, mem_tile, 2, a_ty)
        memA = object_fifo(
            "memA",
            mem_tile,
            compute_tile2,
            2,
            a_ty,
            (
                [
                    (m // r, r * k),
                    (k // s, s),
                    (r, k),
                    (s, 1),
                ]
                if vectorized
                else []
            ),
        )
        object_fifo_link(inA, memA)

        # Input B
        inB = object_fifo("inB", shim_tile, mem_tile, 2, b_ty)

        B_transformations = []
        if vectorized:
            if not b_col_maj:
                B_transformations = [
                    (k // s, s * n),
                    (n // t, t),
                    (s, n),
                    (t, 1),
                ]
            else:
                B_transformations = [
                    (n // t, t * k),
                    (k // s, s),
                    (t, k),
                    (s, 1),
                ]

        memB = object_fifo(
            "memB",
            mem_tile,
            compute_tile2,
            2,
            b_ty,
            B_transformations,
        )
        object_fifo_link(inB, memB)

        # Output C
        memC = object_fifo("memC", compute_tile2, mem_tile, 2, c_ty)
        outC = object_fifo(
            "outC",
            mem_tile,
            shim_tile,
            2,
            c_ty,
            (
                [
                    (m // r, r * n),
                    (r, t),
                    (n // t, r * t),
                    (t, 1),
                ]
                if vectorized
                else []
            ),
        )
        object_fifo_link(memC, outC)

        # Set up a packet-switched flow from core to shim for tracing information
        tiles_to_trace = [compute_tile2]
        if enable_tracing:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

        # Set up compute tiles

        # Compute tile 2
        @core(compute_tile2, f"mm_{m}x{k}x{n}.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
                    elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                    zero(elem_out)

                    for _ in (
                        range_(K_div_k) if K_div_k > 1 else range(1)
                    ):  # issue #1547
                        elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        memA.release(ObjectFifoPort.Consume, 1)
                        memB.release(ObjectFifoPort.Consume, 1)
                    memC.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement

        @runtime_sequence(
            np.ndarray[(A_sz,), np.dtype[dtype_in]],
            np.ndarray[(B_sz,), np.dtype[dtype_in]],
            np.ndarray[(C_sz,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):

            if enable_tracing:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace,
                    shim_tile,
                    trace_size,
                )

            # This example uses only does 2 tile rows to prevent exhaustion of BDs.
            # In general, we do 2-4 at a time to reuse BDs.
            rows_per_block = 2

            # These lists will hold handles to the DMA tasks we configure
            # on the shim. We can later use these handles to start those
            # tasks and wait for their completion.
            a_tasks = []
            b_tasks = []
            c_tasks = []

            A_taps = TensorTiler2D.group_tiler(
                (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
            )
            # There is only one access pattern for B - it tiles the entire matrix in (k x n) tiles.
            if b_col_maj:
                b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k))[0]
            else:
                b_tap = TensorTiler2D.group_tiler(
                    (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
                )[0]
            C_taps = TensorTiler2D.group_tiler(
                (M, N), (m, n), (rows_per_block // 2, N_div_n)
            )
            c_index = 0

            for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
                # that's what this loop is for
                for pingpong in [0, 1]:
                    row_base = (
                        tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                    )
                    num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                    if num_tile_rows <= 0:
                        # At the very last iteration, we may not need a 'pong' iteration
                        break

                    # -- C --
                    # Configure a task on the same channel where the
                    # objectFifo "outC" expects its data to be streamed in
                    # from. Repeat count is how often to repeat this task,
                    # hence for 1 iteration, repeat count is 0. The highest
                    # dimension stride/wrap is applied at every repeat of
                    # the BD. We need to set issue_token=True to be able to
                    # await completion of the task later on using
                    # dma_await_task.

                    c_task = shim_dma_single_bd_task(
                        outC, C, tap=C_taps[c_index], issue_token=True
                    )
                    C_taps.append(C_taps[c_index])
                    c_index += 1
                    dma_start_task(c_task)
                    c_tasks.append(c_task)

                    for tile_row in range(num_tile_rows):
                        # -- A --
                        tile_offset = (row_base + tile_row) % len(A_taps)
                        a_task = shim_dma_single_bd_task(
                            inA, A, tap=A_taps[tile_offset]
                        )
                        A_taps.append(A_taps[tile_offset])
                        dma_start_task(a_task)
                        a_tasks.append(a_task)

                        # -- B --
                        b_task = shim_dma_single_bd_task(inB, B, tap=b_tap)
                        B_taps.append(b_tap)
                        dma_start_task(b_task)
                        b_tasks.append(b_task)

                    if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                        dma_await_task(c_tasks[-2])
                        # Once the task for C has completed, we know that A
                        # and B must have completed as well; free their BDs.
                        dma_free_task(a_tasks[-2])
                        dma_free_task(b_tasks[-2])

            dma_await_task(c_tasks[-1])

            trace_utils.gen_trace_done_aie2(shim_tile)

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )


if __name__ == "__main__":
    main()
