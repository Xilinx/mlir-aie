# dma_transpose/dma_transpose_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern

if len(sys.argv) > 3:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2_1col
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
else:
    raise ValueError("[ERROR] Not enough arguments provided")


def my_passthrough(M, K, N, generate_access_map=False):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]
    data_transform = TensorAccessPattern(
        (M, K), offset=0, sizes=[1, 1, K, M], strides=[1, 1, 1, K]
    )
    if generate_access_map:
        data_transform.visualize(
            show_arrows=True, plot_access_count=False, file_path="transpose_data.png"
        )
        return

    with mlir_mod_ctx() as ctx:

        @device(dev)
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
                in_task = shim_dma_single_bd_task(
                    of_in, A, tap=data_transform, issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, N], issue_token=True
                )

                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("device_name", help="Device name (npu or npu2)", type=str)
    p.add_argument("dims", help="M K", type=int, nargs=2, default=[64, 64])
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
        generate_access_map=args.generate_access_map,
    )
