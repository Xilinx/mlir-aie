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
def my_passthrough(m, k, K):

    # large K must be divisible by small k 
    assert K % k == 0

        
    # send the entire m*K
    big_tile_ty = np.ndarray[(m, K), np.dtype[np.int32]]

    small_tile_ty = np.ndarray[(4, 1), np.dtype[np.int32]]

    
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)


            # AIE-array data movement with object fifos

            # Input
            of_in_shim_to_mem = object_fifo(
                "shim_to_mem",
                ShimTile,
                ComputeTile,
                1,              # single buffer here to allow max buffer transaction 
                big_tile_ty,
            )


            of_out_mem_to_shim = object_fifo(
                "mem_to_shim",
                ComputeTile,
                ShimTile,
                1, 
                small_tile_ty
            )

            # # links of_in to of_out
            object_fifo_link(of_in_shim_to_mem, of_out_mem_to_shim)



            # Compute tile just passes, doesn't do any operation
            @core(ComputeTile)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass
                    
                    

            # set the runtime type as 1D array
            big_runtime_ty = np.ndarray[(m*K,), np.dtype[np.int32]]

            small_runtime_ty = np.ndarray[(4,), np.dtype[np.int32]]
            
            # To/from AIE-array data movement
            @runtime_sequence(big_runtime_ty, big_runtime_ty, small_runtime_ty)
            def sequence(A, B, C):
                
                npu_dma_memcpy_nd(
                    metadata=of_in_shim_to_mem,
                    bd_id=0,
                    mem=A,
                    sizes=[1, 1, 1, m*K],
                    issue_token=True,       # issue_token must be True to wait on the input transmission
                )

                npu_dma_memcpy_nd(
                    metadata=of_out_mem_to_shim, 
                    bd_id=1, 
                    mem=C, 
                    sizes=[1, 1, 1, 4])
                
                # wait only on input to measure the one sided data transmission from DDR
                dma_wait(of_in_shim_to_mem)
                dma_wait(of_out_mem_to_shim)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="m, k, K", type=int, nargs="*")
    args = p.parse_args()

    if len(args.dims) != 3:
        print(
            "ERROR: Must provide all 3 dimensions", file=sys.stderr
        )
        exit(-1)

    my_passthrough(
        m=args.dims[0],
        k=args.dims[1],
        K=args.dims[2],
    )
