#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import argparse
import numpy as np
import tensorflow as tf
import sys

bfloat16 = tf.bfloat16.as_numpy_dtype

from aie.dialects.scf import yield_, for_ as range_
from aie.dialects.aiex import npu_dma_memcpy_nd, npu_sync

from aie.api.dataflow.inout.inout import MyInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.dataflow.objectfifolink import MyObjectFifoLink
from aie.api.kernels.binkernel import BinKernel
from aie.api.phys.device import NPU1Col1
from aie.api.program import MyProgram
from aie.api.worker import MyWorker


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument("-v", "--vectorized", type=bool, default=True)
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    args = argparser.parse_args()
    my_matmul(
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.vectorized,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, dtype_in_str, dtype_out_str, vectorized):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    r = 4
    s = 8
    t = 4
    if dtype_in_str == "i8":
        t = 8
    elif dtype_in_str == "i16":
        s = 4

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    dtype_map = {
        "bf16": bfloat16,
        "i8": np.int8,
        "i16": np.int16,
        "f32": np.float32,
        "i32": np.int32,
    }
    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    num_data_tiles = (M // m) * (N // n)

    # input/output matrices TODO(erika): fix up types
    memref_A_ty = ([M * K], dtype_in)
    memref_B_ty = ([K * N], dtype_in)
    memref_C_ty = ([M * N], dtype_out)

    # submatrices TODO(erika): fix up types
    memref_a_ty = ((m, k), dtype_in)
    memref_b_ty = ((k, n), dtype_in)
    memref_c_ty = ((m, n), dtype_out)

    # AIE Core Function declarations
    scalar_str = "" if vectorized else "scalar_"
    zero = BinKernel(
        f"zero_{scalar_str}{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [memref_c_ty]
    )
    matmul = BinKernel(
        f"matmul_{scalar_str}{dtype_in_str}_{dtype_out_str}",
        f"mm_{m}x{k}x{n}.o",
        [memref_a_ty, memref_b_ty, memref_c_ty],
    )

    inA = MyObjectFifo(2, memref_a_ty)
    memA = MyObjectFifo(
        2,
        memref_a_ty,
        dimensionsToStream=(
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
    inALink = MyObjectFifoLink([inA.second], [memA.first], coords=(0, 1))

    # Input B
    inB = MyObjectFifo(2, memref_b_ty)
    memB = MyObjectFifo(
        2,
        memref_b_ty,
        dimensionsToStream=(
            [
                (k // s, s * n),
                (n // t, t),
                (s, n),
                (t, 1),
            ]
            if vectorized
            else []
        ),
    )
    inBLink = MyObjectFifoLink([inB.second], [memB.first], coords=(0, 1))

    # Output C
    memC = MyObjectFifo(2, memref_c_ty)
    outC = MyObjectFifo(
        2,
        memref_c_ty,
        dimensionsToStream=(
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
    outCLink = MyObjectFifoLink([memC.second], [outC.first], coords=(0, 1))

    def core_fn(a, b, c, zero, matmul):
        for _ in range_(0xFFFFFFFF):
            for _ in (
                range_(num_data_tiles) if num_data_tiles > 1 else range(1)
            ):  # issue #1547
                elem_out = c.acquire(1)
                zero(elem_out)

                for _ in range_(K // k) if (K // k) > 1 else range(1):  # issue #1547
                    elem_in_a = a.acquire(1)
                    elem_in_b = b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    a.release(1)
                    b.release(1)
                    if (K // k) > 1:
                        yield_([])

                c.release(1)
                if num_data_tiles > 1:
                    yield_([])
            yield_([])

    def sequence_fn(A, B, C, inA, inB, outC):
        # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
        rows_per_block = 4
        for tile_row_block in range(ceildiv(M // m, rows_per_block)):
            # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
            # that's what this loop is for
            for pingpong in [0, 1]:
                C_row_offset = (
                    tile_row_block * rows_per_block * m * N
                    + pingpong * rows_per_block // 2 * m * N
                )
                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                bd_id_base = 8 * pingpong
                num_tile_rows = min([rows_per_block // 2, M // m - row_base])
                if num_tile_rows <= 0:
                    # At the very last iteration, we may not need a 'pong' iteration
                    break
                npu_dma_memcpy_nd(
                    metadata=outC.name,
                    bd_id=bd_id_base,
                    mem=C,
                    offsets=[0, 0, 0, C_row_offset],
                    sizes=[num_tile_rows, N // n, m, n],
                    strides=[m * N, n, N, 1],
                )
                for tile_row in range(num_tile_rows):
                    A_row_offset = (row_base + tile_row) * m * K
                    npu_dma_memcpy_nd(
                        metadata=inA.name,
                        bd_id=bd_id_base + 2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset],
                        sizes=[N // n, K // k, m, k],
                        strides=[0, k, K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=inB.name,
                        bd_id=bd_id_base + 2 * tile_row + 2,
                        mem=B,
                        sizes=[N // n, K // k, k, n],
                        strides=[n, k * N, N, 1],
                    )
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    npu_sync(column=0, row=0, direction=0, channel=0)
        npu_sync(column=0, row=0, direction=0, channel=0)

    inout_program = MyInOutProgram(
        sequence_fn,
        [memref_A_ty, memref_B_ty, memref_C_ty],
        [inA.first, inB.first, outC.second],
        coords=(0, 0),
    )

    # AnyMemtile
    c = LogicalCore()
    c2 = c.neighbor()

    worker_program = MyWorker(
        core_fn,
        [memA.second, memB.second, memC.first, zero, matmul],
        AnyCore,  # coords=(0, 2)
    )

    my_program = MyProgram(
        NPU1Col1(),
        worker_programs=[worker_program],
        links=[inALink, inBLink, outCLink],
        inout_program=inout_program,
        placer=SequentialPlace(),  # GraphBasedPlacer() # CoreOnlyPlace() -> anything memtile has to be decided by Programmer
    )

    # g = my_program.get_dataflow_graph()

    my_program.resolve_program()


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
