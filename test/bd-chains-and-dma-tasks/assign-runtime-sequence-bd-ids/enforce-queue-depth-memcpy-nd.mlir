//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=true' %s \
// RUN:   | FileCheck %s
// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=false' %s 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=OFF

// The twin of enforce-queue-depth.mlir on the npu.dma_memcpy_nd path, which is
// the majority of in-tree designs. Both paths push onto the same hardware
// queue and share the poll emitter, so the poll lands against the same
// register and depth bit here: shim DMA_MM2S_Status_0 at 0x1D228 (119336),
// mask 0x400000 (4194304) for a 4-deep queue.

// Only the fifth transfer can overflow. All push forms are checked together
// after memcpy lowering, so the fifth descriptor is written before the poll,
// but its START_QUEUE write must come after the poll.
// CHECK-LABEL: @enforce
// CHECK-COUNT-5: aiex.npu.blockwrite
// CHECK-DAG:     arith.constant 119336 : i32
// CHECK-DAG:     arith.constant 4194304 : i32
// CHECK:         aiex.npu.maskpoll
// CHECK:         aiex.npu.write32
// CHECK-NOT:     aiex.npu.maskpoll

// OFF-LABEL: @enforce
// OFF-NOT:   aiex.npu.maskpoll
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence @enforce(%arg0: memref<1280xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}
