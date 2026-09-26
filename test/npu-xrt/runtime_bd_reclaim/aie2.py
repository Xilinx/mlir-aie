# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Forty fills and forty drains through one shim tile, none of them freed and
# only the last awaited: 80 two-BD tasks over a 16-BD pool. The BD-id pass takes
# ids back from started tasks it can prove finished, so every half-chunk must
# still move from its own offset. A poll that let an id be rewritten while its
# task was queued would move some half-chunk from the wrong offset (or hang).

# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: aie-opt --aie-objectFifo-stateful-transform \
# RUN:   --aie-substitute-shim-dma-allocations \
# RUN:   --aie-assign-runtime-sequence-bd-ids ./aie2.mlir \
# RUN:   | FileCheck %s --check-prefix=MLIR
# RUN: %aiecc --get-xclbin --get-npu-insts --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall -Wextra %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu2% ./test.exe | FileCheck %s --check-prefix=DEVICE
# DEVICE: PASS!

# The pool runs out at the ninth task, before any queue-space poll: the pass
# polls for the oldest fill, which has three fills queued behind it and so is
# done once Task_Queue_Size <= 1 (bits 22:21 clear). Later reclaims also lean on
# the queue-space polls (bit 22) the pass emits from the fifth push on.
# MLIR: arith.constant 6291456 : i32
# MLIR-NEXT: arith.constant 0 : i32
# MLIR-NEXT: arith.constant 119336 : i32
# MLIR-NEXT: aiex.npu.maskpoll

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

N_CHUNKS = 40
# Large enough that a task is still moving data when the command processor
# reaches the next reclaim. With the pass's polls stripped from the lowered
# sequence, this fails every run from chunk 1 on; at 4 KiB chunks it passed
# either way, the DMA outrunning the rewrite.
CHUNK = 16384
LEN = N_CHUNKS * CHUNK


def halves(task, host, i):
    # Each chunk as a chain of two half-chunk BDs, so the pool runs out after
    # eight tasks, before the task queues fill.
    half = CHUNK // 2
    with bds(task) as bd:
        with bd[0]:
            shim_dma_bd(host, offset=i * CHUNK, sizes=[1, 1, 1, half])
            next_bd(bd[1])
        with bd[1]:
            shim_dma_bd(host, offset=i * CHUNK + half, sizes=[1, 1, 1, half])
            EndOp()


def design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            buff_ty = np.ndarray[(LEN,), np.dtype[np.int32]]
            chunk_ty = np.ndarray[(CHUNK,), np.dtype[np.int32]]

            shim = tile(0, 0)
            mem = tile(0, 1)

            of_in = object_fifo("in", shim, mem, 2, chunk_ty)
            of_out = object_fifo("out", mem, shim, 2, chunk_ty)
            object_fifo_link(of_in, of_out)

            @runtime_sequence(buff_ty, buff_ty)
            def sequence(A, B):
                for i in range(N_CHUNKS):
                    fill = dma_configure_task_for(of_in)
                    halves(fill, A, i)
                    dma_start_task(fill)
                    last = i == N_CHUNKS - 1
                    drain = dma_configure_task_for(of_out, issue_token=last)
                    halves(drain, B, i)
                    dma_start_task(drain)
                dma_await_task(drain)

    print(ctx.module)


design()
