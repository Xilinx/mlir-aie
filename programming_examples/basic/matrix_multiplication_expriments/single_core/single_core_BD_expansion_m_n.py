#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.helpers.dialects.ext.scf import _for as range_

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
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
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_matmul(
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
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev, M, K, N, m, k, n, dtype_in_str, dtype_out_str, b_col_maj, trace_size
):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    if dev == "npu":
        if dtype_in_str == "bf16":
            r = 4
            s = 8
            t = 4
        elif dtype_in_str == "i8":
            r = 4
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 4
    else:
        if dtype_in_str == "bf16":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i8":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 8

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

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

    with mlir_mod_ctx() as ctx:

        C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize

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
            matmul_func_name = (
                f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}"
                if not b_col_maj
                else f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}_b_col_maj"
            )
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
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # Set up compute tiles

            # Compute tile 2
            @core(compute_tile2, f"mm_{m}x{k}x{n}.o")
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
                        tiles_to_trace=tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
                        trace_offset=C_sz_in_bytes,
                        ddr_id=2,
                        coretile_events=[
                            # captures input A (PORT_RUNNING_0, at port number 1, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            # captures input B (PORT_RUNNING_1, at port number 2, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            # captures output C (PORT_RUNNING_2, at port number 1, slave for outputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_2,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.CoreEvent.INSTR_EVENT_0,
                            trace_utils.CoreEvent.INSTR_EVENT_1,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # max number of rows tiles to submit each BD for A and B at a time
                # Each BD represents:
                # 1) one row tile for A, i.e., m * K,
                # 2) one col tile for B, i.e., n * K,
                # 3) one out tile for C, i.e., m * n
                max_BDs_per_A_B = 5

                # Number of total row tiles
                total_row_tiles = M // m

                # Number of total col tiles
                total_col_tiles = N // n

                # keep track of the row and col indices
                # for A and B
                A_row_index = 0
                B_col_index = 0

                # counter for initial BD assignment
                initial_BD_cnt = 0

                # flag to indicate when loop should break (see below)
                should_break = False

                # First, submit the initial BDs for each A and B
                for i in range(total_row_tiles):

                    # if col tiles have finished,
                    # start submitting the next row (thus B_col_index = 0)
                    B_col_index = 0

                    for j in range(total_col_tiles):

                        A_row_offset = A_row_index * m * K

                        npu_dma_memcpy_nd(
                            metadata=inA,
                            bd_id=2 * initial_BD_cnt + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_row_offset],
                            sizes=[1, K // k, m, k],
                            strides=[0, k, K, 1],
                        )

                        if not b_col_maj:
                            B_col_offset = B_col_index * n
                            B_sizes = [1, K // k, k, n]
                            B_strides = [0, k * N, N, 1]
                        else:
                            B_col_offset = B_col_index * K * n
                            B_sizes = [1, K // k, n, k]
                            B_strides = [0, k, K, 1]

                        npu_dma_memcpy_nd(
                            metadata=inB,
                            bd_id=2 * initial_BD_cnt + 2,
                            mem=B,
                            offsets=[0, 0, 0, B_col_offset],
                            sizes=B_sizes,
                            strides=B_strides,
                        )

                        # break innermost loop when max BDs reached
                        # or total tiles reached
                        if (initial_BD_cnt == max_BDs_per_A_B - 1) or (
                            initial_BD_cnt == total_row_tiles * total_col_tiles - 1
                        ):
                            should_break = True
                            break

                        B_col_index += 1
                        initial_BD_cnt += 1

                    # break also outermost loop when max BDs reached
                    if should_break:
                        break

                    A_row_index += 1

                # The two loops above will finish either by break or by very low number of row and col tiles.
                # In both cases, the A_row_index and B_col_index variables store the already processed indices.
                # We increase the indices to point to the next row and col tiles.

                # Always increase the col index to point to the next tile
                B_col_index += 1

                # Increase the row index in case we reached all the col tiles
                if B_col_index == total_col_tiles:
                    B_col_index = 0
                    A_row_index += 1

                # modulo counter for A and B, BD IDs
                mod_BD_ID_cnt = 0

                # Second, submit each (m*n) tile for C and then reconfigure BDs for A and B
                # when you know that each tile is done
                for i in range(total_row_tiles):
                    for j in range(total_col_tiles):

                        # i, j for C offsets
                        # A_row_index and B_col_index are used for A and B reconfiguration only
                        C_row_offset = i * m * N
                        C_col_offset = j * n

                        C_offset = C_row_offset + C_col_offset

                        npu_dma_memcpy_nd(
                            metadata=outC,
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[1, 1, m, n],
                            strides=[0, 0, N, 1],
                        )

                        # There are A and B need to be reconfigured when we haven't finished processing the rows
                        # exclude the first time only, i.e., i=0, j=0, so we know we can reconfigure
                        if (i > 0 or j > 0) and (A_row_index < total_row_tiles):

                            A_row_offset = A_row_index * m * K

                            npu_dma_memcpy_nd(
                                metadata=inA,
                                bd_id=2 * mod_BD_ID_cnt + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_row_offset],
                                sizes=[1, K // k, m, k],
                                strides=[0, k, K, 1],
                            )

                            if not b_col_maj:
                                B_col_offset = B_col_index * n
                                B_sizes = [1, K // k, k, n]
                                B_strides = [0, k * N, N, 1]
                            else:
                                B_col_offset = B_col_index * K * n
                                B_sizes = [1, K // k, n, k]
                                B_strides = [0, k, K, 1]

                            npu_dma_memcpy_nd(
                                metadata=inB,
                                bd_id=2 * mod_BD_ID_cnt + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset],
                                sizes=B_sizes,
                                strides=B_strides,
                            )

                            mod_BD_ID_cnt = (mod_BD_ID_cnt + 1) % max_BDs_per_A_B

                            B_col_index += 1
                            if B_col_index == total_col_tiles:
                                B_col_index = 0
                                A_row_index += 1

                        # syncronize for each (m*n) C tile
                        dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
