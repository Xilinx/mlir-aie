//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=true' --split-input-file %s \
// RUN:   | FileCheck %s
// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=false' --split-input-file %s 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=OFF

// aie-unroll-runtime-sequence-loops only unrolls constant-trip loops, so a
// runtime-bound scf.for reaches this pass still rolled. Counting its body once
// would guard only the last syntactic push; the queue carries over the back
// edge, so the first push of the next iteration meets what the previous one
// left behind.

// Three pushes per iteration on a 4-deep queue, never awaited. Every push is
// guarded: from the second iteration onwards the queue is full whichever one
// runs next, so no push in the body is safe by position.
// CHECK-LABEL: @fills_queue
// CHECK:         scf.for
// CHECK-COUNT-3: aiex.npu.maskpoll

// OFF-LABEL: @fills_queue
// OFF-NOT:   aiex.npu.maskpoll
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence @fills_queue(%arg0: memref<1280xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<1280xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc0} : memref<1280xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc0} : memref<1280xi32>
    }
  }
}

// -----

// A body that drains what it starts needs no poll at all. The lowering gives
// every S2MM push a token, and the wait pops that token plus everything queued
// ahead of it, so occupancy returns to empty every iteration and the fixed
// point settles without the queue ever filling.
// CHECK-LABEL: @drains_queue
// CHECK-NOT:   aiex.npu.maskpoll
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc_in (%tile_0_0, S2MM, 0)
  aie.runtime_sequence @drains_queue(%arg0: memref<1280xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc_in} : memref<1280xi32>
      aiex.npu.dma_wait { symbol = @alloc_in }
    }
  }
}
