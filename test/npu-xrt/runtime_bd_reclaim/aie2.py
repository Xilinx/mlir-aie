# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Forty chunks through one shim tile, each as two half-chunk fills and two
# half-chunk drains on two lanes, none of them freed until the end and only the
# last drains awaited: 160 single-BD tasks over a 16-BD pool. The BD-id pass
# takes ids back from started tasks it can prove finished, so every half-chunk
# must still move from its own offset. A poll that let an id be rewritten while
# its task was queued would move some half-chunk from the wrong offset (or
# hang).

# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform \
# RUN:   --aie-substitute-shim-dma-allocations \
# RUN:   --aie-assign-runtime-sequence-bd-ids='reclaim-bds=true' ./aie2.mlir \
# RUN:   | FileCheck %s --check-prefix=MLIR
# RUN: %aiecc --reclaim-runtime-bds --get-xclbin --get-npu-insts --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall -Wextra %xrt_flags %host_link_flags %test_utils_flags
# RUN: %run_on_npu2% ./test.exe | FileCheck %s --check-prefix=DEVICE
# DEVICE: PASS!

# Four queues of four fill the pool, so it runs out at the 17th task, before
# any queue-space poll: the pass polls for the oldest fill, which has three
# fills queued behind it and so is done once Task_Queue_Size <= 1 (bits 22:21
# clear). Later reclaims also lean on the queue-space polls (bit 22) the pass
# emits from the fifth push on each queue.
# MLIR: arith.constant 6291456 : i32
# MLIR-NEXT: arith.constant 0 : i32
# MLIR-NEXT: arith.constant 119336 : i32
# MLIR-NEXT: aiex.npu.maskpoll

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2, Tile

N_CHUNKS = 40
# Large enough that a task is still moving data when the command processor
# reaches the next reclaim; at 4 KiB chunks the DMA outruns the rewrite.
CHUNK = 16384
HALF = CHUNK // 2
LEN = N_CHUNKS * CHUNK
LANES = 2


def design():
    buff_ty = np.ndarray[(LEN,), np.dtype[np.int32]]
    half_ty = np.ndarray[(HALF,), np.dtype[np.int32]]

    shim, mem = Tile(0, 0), Tile(0, 1)
    ins = [ObjectFifo(half_ty, depth=2, name=f"in{h}") for h in range(LANES)]
    outs = [f.cons().forward(tile=mem, name=f"out{h}") for h, f in enumerate(ins)]

    def sequence(a, b, *ends):
        fills, drains = ends[:LANES], ends[LANES:]
        for i in range(N_CHUNKS):
            for h in range(LANES):
                at = dict(offset=i * CHUNK + h * HALF, sizes=[1, 1, 1, HALF])
                fills[h].fill(a, **at)
                drains[h].drain(b, wait=i == N_CHUNKS - 1, **at)

    rt = Runtime(
        sequence,
        [buff_ty, buff_ty]
        + [f.prod(tile=shim) for f in ins]
        + [f.cons(tile=shim) for f in outs],
    )
    return Program(NPU2(), rt).resolve_program()


print(design())
