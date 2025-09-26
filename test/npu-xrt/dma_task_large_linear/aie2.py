#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai_npu1, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: clang %S/test.cpp -o test -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu1% ./test
import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

dtype = np.int32  # TODO: change to int8
BUFFER_LEN = 65536
BUFFER_TILE_LEN = BUFFER_LEN // (2**5)


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            buff_ty = np.ndarray[(BUFFER_LEN,), np.dtype[dtype]]
            buff_tile_ty = np.ndarray[(BUFFER_TILE_LEN,), np.dtype[dtype]]

            ShimTile = tile(0, 0)
            ComputeTile = tile(0, 2)

            of_in = object_fifo("in", ShimTile, ComputeTile, 2, buff_tile_ty)
            of_out = object_fifo("out", ComputeTile, ShimTile, 2, buff_tile_ty)

            @core(ComputeTile)
            def core_body():
                for _ in range_(0, 0xFFFFFFFF):
                    for _tile in range_(BUFFER_LEN // BUFFER_TILE_LEN):
                        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)

                        for i in range_(BUFFER_TILE_LEN):
                            elem_out[i] = elem_in[i] + 1

                        of_in.release(ObjectFifoPort.Consume, 1)
                        of_out.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            @runtime_sequence(buff_ty, buff_ty)
            def sequence(A, B):
                in_act_task = dma_configure_task_for(of_in, issue_token=True)
                with bds(in_act_task) as bd:
                    with bd[0]:
                        shim_dma_bd(A, sizes=[1, 1, 1, BUFFER_LEN])
                        EndOp()

                out_task = dma_configure_task_for(of_out, issue_token=True)
                with bds(out_task) as bd:
                    with bd[0]:
                        shim_dma_bd(B, sizes=[1, 1, 1, BUFFER_LEN])
                        EndOp()

                dma_start_task(in_act_task, out_task)
                dma_await_task(in_act_task, out_task)

    print(ctx.module)


design()
