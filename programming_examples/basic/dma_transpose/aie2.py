# dma_transpose/aie2.py -*- Python -*-
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
from aie.helpers.tensortiler.tensortiler2d import TensorTile


def my_passthrough(M, K, N, generate_acccess_map=False):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]
    data_transform = TensorTile(
        tensor_height=M,
        tensor_width=K,
        sizes=[1, K, M, 1],
        strides=[1, 1, K, 1],
        offset=0,
    )
    if generate_acccess_map:
        data_transform.visualize(
            plot_access_count=False, file_path="transpose_data.png"
        )
        return

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tensor_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tensor_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                # The strides below are configured to read across all rows in the same column
                # Stride of K in dim/wrap 2 skips an entire row to read a full column
                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=1,
                    mem=A,
                    tensor_tile=data_transform,
                    issue_token=True,
                )
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
                dma_wait(of_in, of_out)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="M K", type=int, nargs="*", default=[64, 64])
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="Produce a file showing data access order",
    )
    args = p.parse_args()

    if len(args.dims) != 2:
        print(
            "ERROR: Must provide either no dimensions or both M and K", file=sys.stderr
        )
        exit(-1)
    my_passthrough(
        M=args.dims[0],
        K=args.dims[1],
        N=args.dims[0] * args.dims[1],
        generate_acccess_map=args.generate_access_map,
    )
