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


# this resembles the buffer A data layout and transformations
def my_passthrough(m, M, k, K, n, N, data_layout_DDR):


    # large M must be divisible by small m 
    assert M % m == 0

    # large K must be divisible by small k 
    assert K % k == 0

    # large N must be divisible by small n
    assert N % n == 0

    
    mem_tile_A_ty = np.ndarray[(m, k), np.dtype[np.int32]]

    mem_tile_B_ty = np.ndarray[(k, n), np.dtype[np.int32]]
    
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)


            # AIE-array data movement with object fifos

            # Input
            of_in_shim_to_mem_A = object_fifo(
                "shim_to_mem_A",
                ShimTile,
                MemTile,
                2,              # double buffer 
                mem_tile_A_ty,
            )


            of_out_mem_A_to_shim = object_fifo(
                "mem_A_to_shim",
                MemTile,
                ShimTile,
                2, 
                mem_tile_A_ty
            )

            # links of_in to of_out
            object_fifo_link(of_in_shim_to_mem_A, of_out_mem_A_to_shim)


            # Input
            of_in_shim_to_mem_B = object_fifo(
                "shim_to_mem_B",
                ShimTile,
                MemTile,
                2,              # double buffer 
                mem_tile_B_ty,
            )


            of_out_mem_B_to_shim = object_fifo(
                "mem_B_to_shim",
                MemTile,
                ShimTile,
                2, 
                mem_tile_B_ty
            )

            # links of_in to of_out
            object_fifo_link(of_in_shim_to_mem_B, of_out_mem_B_to_shim)



            # Compute tile just passes, doesn't do any operation
            @core(ComputeTile)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass
                    
                    

            # set the runtime type as 1D array.
            # send the entire A buffer of M*K size
            runtime_A_ty = np.ndarray[(M*K,), np.dtype[np.int32]]

            runtime_B_ty = np.ndarray[(K*N,), np.dtype[np.int32]]

            runtime_C_ty = np.ndarray[(M*K + K*N,), np.dtype[np.int32]]

            
            # To/from AIE-array data movement
            @runtime_sequence(runtime_A_ty, runtime_B_ty, runtime_C_ty)
            def sequence(A, B, C):
                
                npu_dma_memcpy_nd(
                    metadata=of_in_shim_to_mem_A,
                    bd_id=0,
                    mem=A,
                    sizes=([M//m, K//k, m, k] if data_layout_DDR=="blocked" else 
                           [1, 1, 1, M*K]),
                    strides=([m*K, k, K, 1] if data_layout_DDR=="blocked" else 
                             None),
                )

                npu_dma_memcpy_nd(
                    metadata=of_out_mem_A_to_shim, 
                    bd_id=1, 
                    mem=C,
                    sizes=([M//m, K//k, m, k] if data_layout_DDR=="blocked" else 
                           [1, 1, 1, M*K]),
                    strides=([m*K, k, K, 1] if data_layout_DDR=="blocked" else
                             None),
                )

                npu_dma_memcpy_nd(
                    metadata=of_in_shim_to_mem_B,
                    bd_id=2,
                    mem=B,
                    sizes=([N//n, K//k, k, n] if data_layout_DDR=="blocked" else
                           [1, 1, 1, K*N]),
                    strides=([n, k*N, N, 1] if data_layout_DDR=="blocked" else
                             None),
                )

                npu_dma_memcpy_nd(
                    metadata=of_out_mem_B_to_shim, 
                    bd_id=3, 
                    mem=C,
                    offsets=[0, 0, 0, M*K],  # offset for the A output buffer of M*K size
                    sizes=([N//n, K//k, k, n] if data_layout_DDR=="blocked" else
                           [1, 1, 1, K*N]),
                    strides=([n, k*N, N, 1] if data_layout_DDR=="blocked" else
                             None),
                )
                
                
                dma_wait(of_out_mem_A_to_shim)
                dma_wait(of_out_mem_B_to_shim)



    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="m, M, k, K, n, N", type=int, nargs="*")
    p.add_argument("data_layout_DDR", help="Layout of data in DDR", type=str, choices=["linear", "blocked"], default="linear")
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
        data_layout_DDR=args.data_layout_DDR,
    )
