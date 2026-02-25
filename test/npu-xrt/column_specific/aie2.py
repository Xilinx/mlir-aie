# column_specific/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
#
# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %run_on_npu1% ./test.exe -x final.xclbin -k MLIR_AIE -i insts.bin

import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_3col
col = 2
line_size = 1024


def my_passthrough():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, line_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, line_ty)
            object_fifo_link(of_in, of_out)

            # To/from AIE-array data movement
            @runtime_sequence(vector_ty, vector_ty, vector_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N], issue_token=True
                )
                npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
                dma_wait(of_in, of_out)

    print(ctx.module)


my_passthrough()
