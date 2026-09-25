# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A fill with a 6-dimension access pattern, more than a BD's 4, run for two
# passes. Its three iteration dimensions do not merge, so the compiler splits
# off the outer two into 6 pieces of [4, 2, 8, 16], one task each: more than
# the channel queues (4), so it has to wait for queue space as it pushes them,
# and each piece is started once per pass. The drain copies what the fill
# gathers, in the order it gathers it, into a contiguous buffer: a wrong piece
# offset, a wrong per-piece repeat count, a piece out of order or a pass that
# does not start from the first index shows up as a mismatch.

# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: aie-opt --aie-objectFifo-stateful-transform \
# RUN:   --aie-substitute-shim-dma-allocations \
# RUN:   --aie-decompose-large-dma-bd ./aie2.mlir \
# RUN:   | FileCheck %s --check-prefix=MLIR
# RUN: %aiecc --get-xclbin --get-npu-insts --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall -Wextra %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu2% ./test.exe | FileCheck %s --check-prefix=DEVICE
# DEVICE: PASS!

# 6 pieces of 4 executions at 16384 * i + 4096 * j, each restarted for pass 2.
# MLIR:          offset = 0 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR-NEXT:       aie.end
# MLIR-NEXT:     } {repeat_count = 3 : i32}
# MLIR:          offset = 4096 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR:          offset = 8192 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR:          offset = 16384 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR:          offset = 20480 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR:          offset = 24576 len = 256 sizes = [4, 2, 8, 16] strides = [16, 1024, 128, 1])
# MLIR:          aiex.dma_start_task
# MLIR-COUNT-6:  aiex.dma_start_task
# MLIR-NEXT:     aiex.dma_await_task

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

LEN = 32768
SIZES = [2, 3, 4, 2, 8, 16]
STRIDES = [16384, 4096, 16, 1024, 128, 1]
PASSES = 2
EXECUTIONS = int(np.prod(SIZES[:-3]))
CHUNK = int(np.prod(SIZES[-3:]))


def design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            buff_ty = np.ndarray[(LEN,), np.dtype[np.int32]]
            obj_ty = np.ndarray[(CHUNK,), np.dtype[np.int32]]

            shim = tile(0, 0)
            mem = tile(0, 1)

            of_in = object_fifo("in", shim, mem, 2, obj_ty)
            of_out = object_fifo("out", mem, shim, 2, obj_ty)
            object_fifo_link(of_in, of_out)

            @runtime_sequence(buff_ty, buff_ty)
            def sequence(A, B):
                fill = dma_configure_task_for(
                    of_in, repeat_count=PASSES * EXECUTIONS - 1
                )
                with bds(fill) as bd:
                    with bd[0]:
                        shim_dma_bd(A, sizes=SIZES, strides=STRIDES)
                        EndOp()
                dma_start_task(fill)
                drain = shim_dma_single_bd_task(
                    of_out,
                    B,
                    sizes=[1, 1, 1, PASSES * EXECUTIONS * CHUNK],
                    issue_token=True,
                )
                dma_start_task(drain)
                dma_await_task(drain)
                dma_free_task(fill)

    print(ctx.module)


design()
