#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
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
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
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
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, dtype_in_str, dtype_out_str, trace_size):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

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

    
    # Matrix dimensions    
    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    # used for sizes in npu_dma_memcpy_nd()
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n

    # used for strides in npu_dma_memcpy_nd()
    m_x_K = m * K
    k_x_N = k * N
    m_x_N = m * N

    # tiles in M and N dimensions
    tiles = M_div_m * N_div_n


    with mlir_mod_ctx() as ctx:

        C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize

        @device(AIEDevice.npu1_1col)
        def device_body():
            a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
            b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
            c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]


            mem_a_ty = np.ndarray[(m, K), np.dtype[dtype_in]]
            mem_b_ty = np.ndarray[(K, n), np.dtype[dtype_in]]



            # AIE Core Function declarations
            func_type = "" if vectorized else "scalar_"
            zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[c_ty])
            matmul = external_func(
                f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}",
                inputs=[a_ty, b_ty, c_ty],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input A

            # memTile takes entire m x K block in row-major order from DDR
            # and transforms it into blocks of m x k (using 3D tranformation of shimTile)
            inA = object_fifo(
                "inA", 
                shim_tile, 
                mem_tile, 
                2, 
                mem_a_ty,
                # None,
                # [
                #     [
                #         (K_div_k, k), 
                #         (m, K), 
                #         (k, 1),
                #     ]
                # ],
            )

            # computeTile takes m x k blocks
            memA = object_fifo(
                "memA",
                mem_tile,
                compute_tile2,
                2,
                a_ty,          
                [
                    # (K_div_k, m*k),
                    # ((k//s)*(m//r), s), 
                    (m // r, r * k),
                    (k // s, s),
                    (r, k), 
                    (s, 1),
                ],
            )
            object_fifo_link(inA, memA)


            # Input B

            # No transformation needed here because data already transformed using npu_dma_memcpy_nd()
            inB = object_fifo("inB", shim_tile, mem_tile, 2, mem_b_ty)

            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile2,
                2,
                b_ty,
                [
                    # (K_div_k, k*n),
                    # ((n//t)*(k//s), t),
                    (k // s, s * n),
                    (n // t, t),
                    (s, n),
                    (t, 1),
                ],
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
                        tiles_to_trace,
                        shim_tile,
                        trace_size,
                        C_sz_in_bytes,
                        events=[
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


                npu_dma_memcpy_nd(
                    metadata=inA,
                    bd_id=0,
                    mem=A,
                    offsets=[0, 0, 0, 0],
                    sizes=[N_div_n, K_div_k, m, k],
                    strides=[0, k, K, 1],
                )
        
                npu_dma_memcpy_nd(
                    metadata=inB,
                    bd_id=1,
                    mem=B,
                    sizes=[N_div_n, K_div_k, k, n],
                    strides=[n, k_x_N, N, 1],
                )


                npu_dma_memcpy_nd(
                    metadata=outC,
                    bd_id=2,
                    mem=C,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, N_div_n, m, n],
                    strides=[m_x_N, n, N, 1],
                    )

                dma_wait(outC)



                # m_K_tiles = 5

                # # assume for now that m_tiles are divisible by m_K_tiles
                # M_div_m_div_m_K_tiles = M_div_m // m_K_tiles

                # base_bd_id = 0

                # for row_iter_m_K in range(M_div_m_div_m_K_tiles):

                #     for iter in range(m_K_tiles):

                #         # offset increases in the M dimension 
                #         A_offset = m_x_K * (row_iter_m_K * m_K_tiles + iter)

                #         # assign BDs
                #         bd_id_A = base_bd_id + iter
                #         bd_id_B = base_bd_id + m_K_tiles + iter


                #         # npu_dma_memcpy_nd(
                #         #     metadata=inA,
                #         #     bd_id=bd_id_A,
                #         #     mem=A,
                #         #     offsets=[0, 0, 0, A_offset],
                #         #     sizes=[N_div_n, 1, m, K],
                #         #     strides=[0, 0, K, 1],
                #         # )



                #         # npu_dma_memcpy_nd(
                #         #     metadata=inB,
                #         #     bd_id=bd_id_B,
                #         #     mem=B,
                #         #     sizes=[1, N_div_n, K, n],
                #         #     strides=[0, n, N, 1],
                #         # )


                #         npu_dma_memcpy_nd(
                #             metadata=inA,
                #             bd_id=bd_id_A,
                #             mem=A,
                #             offsets=[0, 0, 0, A_offset],
                #             sizes=[N_div_n, K_div_k, m, k],
                #             strides=[0, k, K, 1],
                #         )



                #         npu_dma_memcpy_nd(
                #             metadata=inB,
                #             bd_id=bd_id_B,
                #             mem=B,
                #             sizes=[N_div_n, K_div_k, k, n],
                #             strides=[n, k_x_N, N, 1],
                #         )


                #     C_offset = m_x_N * (row_iter_m_K * m_K_tiles)

                #     bd_id_C = base_bd_id + 2*m_K_tiles

                #     npu_dma_memcpy_nd(
                #         metadata=outC,
                #         bd_id=bd_id_C,
                #         mem=C,
                #         offsets=[0, 0, 0, C_offset],
                #         sizes=[m_K_tiles, N_div_n, m, n],
                #         strides=[m_x_N, n, N, 1],
                #     )


                    # dma_wait(outC)


    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)