# test/npu-xrt/objectfifo_repeat/init_values_repeat/aie2.py
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai, peano
#
# RUN: %python %S/aie2.py 4096 > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --no-xchesscc --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu ./test.exe -x final.xclbin -i insts.txt -k MLIR_AIE -l 4096 | FileCheck %s
# CHECK: PASS!
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

N = 4096
dev = AIEDevice.npu1_1col
col = 0
memtile_repeat_count = 4

if len(sys.argv) > 1:
    N = int(sys.argv[1])
data_out_size = N * memtile_repeat_count

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def init_values_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
            tensor_out_ty = np.ndarray[(data_out_size,), np.dtype[np.int32]]

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile = tile(col, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo(
                "in",
                MemTile,
                ComputeTile,
                1,
                tensor_ty,
                initValues=[
                    np.arange(1, N + 1, dtype=np.int32),
                ],
            )
            of_in.set_repeat_count(memtile_repeat_count)
            of_out = object_fifo("out", ComputeTile, ShimTile, 1, tensor_ty)

            # Compute tile
            @core(ComputeTile)
            def core_body():
                # Effective while(1)
                for _ in range_(sys.maxsize):
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(N):
                        elem_out[i] = elem_in[i]
                    of_out.release(ObjectFifoPort.Produce, 1)
                    of_in.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty, tensor_out_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=0,
                    mem=C,
                    sizes=[1, 1, 1, data_out_size],
                )
                # of_out will only complete after of_in completes, so we just wait on of_out instead of both
                dma_wait(of_out)

    print(ctx.module)


init_values_repeat()
