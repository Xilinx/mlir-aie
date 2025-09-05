#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
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
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
    )


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

    with mlir_mod_ctx() as ctx:

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
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # The stack size choice is an important choice!
            # The Peano compiler uses a stack size in this kernel greater than the default one
            # (default is 0x400, chess' stack size is smaller).
            # Exceding the stack size leads to wrong results from the kernel, but no error is triggered.
            # Stack usage can be checked as explained here:
            # https://github.com/Xilinx/llvm-aie/issues/487#issuecomment-2969438585
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
                        tiles_to_trace=tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
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

                # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
                rows_per_block = 4
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
                    # that's what this loop is for
                    for pingpong in [0, 1]:
                        C_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            # At the very last iteration, we may not need a 'pong' iteration
                            break
                        npu_dma_memcpy_nd(
                            metadata=outC,
                            bd_id=bd_id_base,
                            mem=C,
                            offsets=[0, 0, 0, C_row_offset],
                            sizes=[num_tile_rows, N // n, m, n],
                            strides=[m * N, n, N, 1],
                        )
                        for tile_row in range(num_tile_rows):
                            A_row_offset = (row_base + tile_row) * m * K
                            npu_dma_memcpy_nd(
                                metadata=inA,
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_row_offset],
                                sizes=[N // n, K // k, m, k],
                                strides=[0, k, K, 1],
                            )

                            if not b_col_maj:
                                B_sizes = [N // n, K // k, k, n]
                                B_strides = [n, k * N, N, 1]
                            else:
                                B_sizes = [N // n, K // k, n, k]
                                B_strides = [n * K, k, K, 1]

                            npu_dma_memcpy_nd(
                                metadata=inB,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                sizes=B_sizes,
                                strides=B_strides,
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outC)
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
