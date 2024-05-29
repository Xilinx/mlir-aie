# passthrough_dmas/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# REQUIRES: ryzen_ai

# RUN: %python %s > ./aie.mlir
# RUN: %python aiecc.py --no-aiesim --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt ./aie.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++11 -Wall %xrt_flags -lrt -lstdc++ -lboost_program_options -lboost_filesystem
# RUN: %run_on_npu ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.txt | FileCheck %s
# CHECK: PASS!


import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_1col
col = 0

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])

TILE_SIZE = 1024


def txn_basic():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(TILE_SIZE, T.i32())

            ShimTile = tile(col, 0)
            ComputeTile2 = tile(col, 2)

            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, memRef_ty)

            # write constant 42 to the output fifo
            @core(ComputeTile2)
            def core_body():
                for _ in for_(1000000):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(TILE_SIZE):
                        v1 = arith.constant(42, T.i32())
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            tensor_ty = T.memref(N, T.i32())

            @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


txn_basic()
