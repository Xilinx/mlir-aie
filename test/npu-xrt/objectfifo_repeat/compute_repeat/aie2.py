# test/npu-xrt/objectfifo_repeat/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# REQUIRES: ryzen_ai, valid_xchess_license
#
# RUN: %python %S/aie2.py 4096 > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe -x final.xclbin -i insts.txt -k MLIR_AIE -l 4096 | FileCheck %s
# CHECK: PASS!
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_1col
col = 0
repeat_count = 4

if len(sys.argv) > 1:
    N = int(sys.argv[1])
data_out_size = N * repeat_count

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def compute_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
            tensor_out_ty = np.ndarray[(data_out_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            ComputeTile = tile(col, 3)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile, 1, tensor_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 1, tensor_ty)
            of_out.set_repeat_count(repeat_count)
            object_fifo_link(of_in, of_out)

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty, tensor_out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=0,
                    mem=C,
                    sizes=[1, 1, 1, data_out_size],
                )
                # of_out will only complete after of_in completes, so we just wait on of_out instead of both
                dma_wait(of_out)

    print(ctx.module)


compute_repeat()
