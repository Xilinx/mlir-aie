#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

# REQUIRES: ryzen_ai, peano
#
# RUN: %run_on_npu1% %PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG -I %aie_runtime_lib%/AIE2 -c %S/kernel.cc -o kernel.o
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %python aiecc.py --no-aiesim --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin

# ObjectFIFO lock C API test with depth 3 (triple buffering).

import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *

N = 1024
NUM_ITER = 9  # multiple of 3 for clean triple-buffer cycling
tile_ty = np.ndarray[(N,), np.dtype[np.int32]]


def design():

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            shim_tile = tile(0, 0)
            compute_tile = tile(0, 2)

            # Depth 3: triple buffering
            of_in = object_fifo("of_in", shim_tile, compute_tile, 3, tile_ty)
            of_out = object_fifo("of_out", compute_tile, shim_tile, 3, tile_ty)

            scale_fn = external_func(
                "scale_kernel",
                inputs=[
                    tile_ty,  # in buffer 0
                    tile_ty,  # in buffer 1
                    tile_ty,  # in buffer 2
                    tile_ty,  # out buffer 0
                    tile_ty,  # out buffer 1
                    tile_ty,  # out buffer 2
                    T.index(),  # in acq_lock
                    T.index(),  # in rel_lock
                    T.index(),  # out acq_lock
                    T.index(),  # out rel_lock
                ],
            )

            @core(compute_tile, "kernel.o")
            def core_body():
                in_buf0 = of_in.get_buffer(0)
                in_buf1 = of_in.get_buffer(1)
                in_buf2 = of_in.get_buffer(2)
                in_acq, in_rel = of_in.get_lock(ObjectFifoPort.Consume)

                out_buf0 = of_out.get_buffer(0)
                out_buf1 = of_out.get_buffer(1)
                out_buf2 = of_out.get_buffer(2)
                out_acq, out_rel = of_out.get_lock(ObjectFifoPort.Produce)

                scale_fn(
                    in_buf0,
                    in_buf1,
                    in_buf2,
                    out_buf0,
                    out_buf1,
                    out_buf2,
                    in_acq,
                    in_rel,
                    out_acq,
                    out_rel,
                )

            @runtime_sequence(
                np.ndarray[(N * NUM_ITER,), np.dtype[np.int32]],
                np.ndarray[(N * NUM_ITER,), np.dtype[np.int32]],
            )
            def sequence(inTensor, outTensor):
                npu_dma_memcpy_nd(
                    metadata=of_out,
                    bd_id=1,
                    mem=outTensor,
                    sizes=[1, 1, 1, N * NUM_ITER],
                    issue_token=True,
                )
                npu_dma_memcpy_nd(
                    metadata=of_in,
                    bd_id=0,
                    mem=inTensor,
                    sizes=[1, 1, 1, N * NUM_ITER],
                )
                dma_wait(of_out)

        print(ctx.module)


design()
