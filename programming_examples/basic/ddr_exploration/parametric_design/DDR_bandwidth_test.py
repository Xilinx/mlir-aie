# dma_transpose/dma_transpose.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern



# npu_cols represent the number of npu colums

def my_passthrough(m, M, k, K, n, N, npu_cols, data_layout_DDR):


    # large M must be divisible by small (m * npu_cols) 
    assert M % (npu_cols * m) == 0

    # large K must be divisible by small k 
    assert K % k == 0

    # large N must be divisible by small (n * npu_cols)
    assert N % (npu_cols * n) == 0


    mem_tile_A_ty = np.ndarray[(m, k), np.dtype[np.int8]]

    mem_tile_B_ty = np.ndarray[(k, n), np.dtype[np.int8]]
    
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():

            # Tile declarations
            ShimTiles = [tile(i, 0) for i in range(npu_cols)]
            MemTiles = [tile(i, 1) for i in range(npu_cols)]
            
            # declares OFs in for A buffers
            of_ins_A = [
                object_fifo(f"inA{i}", ShimTiles[i], MemTiles[i], 2, mem_tile_A_ty) 
                for i in range(npu_cols)
            ]

            # declares OFs in for B buffers
            of_ins_B = [
                object_fifo(f"inB{i}", ShimTiles[i], MemTiles[i], 2, mem_tile_B_ty) 
                for i in range(npu_cols)
            ]

            of_outs_A = [
                object_fifo(f"outA{i}", MemTiles[i], ShimTiles[i], 2, mem_tile_A_ty) 
                for i in range(npu_cols)
            ]

            of_outs_B = [
                object_fifo(f"outB{i}", MemTiles[i], ShimTiles[i], 2, mem_tile_B_ty) 
                for i in range(npu_cols)
            ]


            # link OFs in with OFs out
            for i in range(npu_cols):

                # link A
                object_fifo_link(of_ins_A[i], of_outs_A[i])

                # link B
                object_fifo_link(of_ins_B[i], of_outs_B[i])

            
        

            # A buffer size
            runtime_A_ty = np.ndarray[(M*K,), np.dtype[np.int8]]
                    
            # B buffer size
            runtime_B_ty = np.ndarray[(K*N,), np.dtype[np.int8]]

            # C buffer size, which is the summation of both A and B
            runtime_C_ty = np.ndarray[(M*K + K*N,), np.dtype[np.int8]]

            
            # To/from AIE-array data movement
            @runtime_sequence(runtime_A_ty, runtime_B_ty, runtime_C_ty)
            def sequence(A, B, C):


                for i in range(npu_cols):

                    
                    # A is always in row-major
                    sizes_A_row_maj = [M//m//npu_cols, K//k, m, k]
                    strides_A_row_maj = [m*K*npu_cols, k, K, 1]

                    # inputs A
                    npu_dma_memcpy_nd(
                        metadata=of_ins_A[i],
                        bd_id=0,        # bd_id = 0 for each column
                        mem=A,
                        offsets=([0, 0, 0, m*K*i]),
                        sizes=(sizes_A_row_maj),
                        strides=(strides_A_row_maj),
                    )

                    # outputs A
                    npu_dma_memcpy_nd(
                        metadata=of_outs_A[i], 
                        bd_id=2, 
                        mem=C,
                        offsets=([0, 0, 0, m*K*i]),
                        sizes=(sizes_A_row_maj),
                        strides=(strides_A_row_maj),
                    )

                    # inputs B
                    # row-major
                    sizes_B_row_maj = [N//n//npu_cols, K//k, k, n]
                    strides_B_row_maj = [n*npu_cols, k*N, N, 1]

                    # col-major
                    sizes_B_col_maj = [N//n//npu_cols, K//k, n, k]
                    strides_B_col_maj = [K*n*npu_cols, k, K, 1]

                    npu_dma_memcpy_nd(
                        metadata=of_ins_B[i],
                        bd_id=1,
                        mem=B,
                        offsets=([0, 0, 0, K*n*i]),
                        sizes=(sizes_B_row_maj if data_layout_DDR=="A_row_B_row" else
                                sizes_B_col_maj),
                        strides=(strides_B_row_maj if data_layout_DDR=="A_row_B_row" else
                                strides_B_col_maj),
                    )

                    

                    # outputs B
                    npu_dma_memcpy_nd(
                        metadata=of_outs_B[i], 
                        bd_id=3, 
                        mem=C,
                        offsets=([0, 0, 0, M*K + K*n*i]),   # A writes in first M*K and B after that
                        sizes=(sizes_B_row_maj if data_layout_DDR=="A_row_B_row" else
                                sizes_B_col_maj),
                        strides=(strides_B_row_maj if data_layout_DDR=="A_row_B_row" else
                                strides_B_col_maj),
                    )
                

                # wait for the outputs after BD programming
                dma_wait(*of_outs_A)
                dma_wait(*of_outs_B)



    print(ctx.module)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="m, M, k, K, n, N", type=int, nargs="*")
    p.add_argument("npu_cols", help="Number of NPU columns", type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument("data_layout_DDR", help="Layout of data in DDR", type=str, choices=["A_row_B_row", "A_row_B_col"], default="A_row_B_row")
    args = p.parse_args()

    if len(args.dims) != 6:
        print(
            "ERROR: Must provide all 6 dimensions", file=sys.stderr
        )
        exit(-1)

    my_passthrough(
        m=args.dims[0],
        M=args.dims[1],
        k=args.dims[2],
        K=args.dims[3],
        n=args.dims[4],
        N=args.dims[5],
        npu_cols=args.npu_cols,
        data_layout_DDR=args.data_layout_DDR,
    )
