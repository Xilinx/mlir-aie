# vector_vector_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# REQUIRES: ryzen_ai, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --no-xchesscc --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %S/aie.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++11 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.txt | FileCheck %s
# CHECK: PASS!
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


def my_vector_add():
    N = 256
    n = 16
    N_div_n = N // n

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

        # AIE Core Function declarations

        # Tile declarations
        ShimTile = tile(int(sys.argv[2]), 0)
        MemTile = tile(int(sys.argv[2]), 1)
        ComputeTile2 = tile(int(sys.argv[2]), 2)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo(
            "in1",
            MemTile,
            ComputeTile2,
            1,
            tensor_ty,
            initValues=[
                np.arange(1, N + 1, dtype=np.int32),
            ],
        )
        of_in2 = object_fifo("in2", ShimTile, ComputeTile2, buffer_depth, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, tile_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                elem_in1 = of_in1.acquire(ObjectFifoPort.Consume, 1)
                for j in range_(N_div_n):
                    elem_in2 = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(n):
                        elem_out[i] = elem_in1[j * N_div_n + i] + elem_in2[i]
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_in1.release(ObjectFifoPort.Consume, 1)

        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
        def sequence(A, B, C):
            npu_dma_memcpy_nd(metadata=of_in2, bd_id=2, mem=B, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, N])
            # of_out will only complete after of_in completes, so we just wait on of_out instead of both
            dma_wait(of_out)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
