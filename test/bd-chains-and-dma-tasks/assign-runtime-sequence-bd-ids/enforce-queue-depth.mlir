//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=true' %s \
// RUN:   | FileCheck %s
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s --check-prefix=OFF

// With enforcement on, a push that could land on a full task queue is preceded
// by a poll of the channel's live occupancy, so room becomes a precondition
// instead of a bet on drain timing. Whether the queue actually fills depends on
// how fast the consumer drains -- HW-verified on Strix, the same eight pushes
// set Task_Queue_Overflow behind a stalled consumer and leave it clear behind a
// healthy one -- which is exactly why it cannot be decided at compile time.
//
// The poll waits for the depth bit of Task_Queue_Size to be clear. For a 4-deep
// queue that is bit 22, i.e. mask 0x400000 (4194304) and value 0, against shim
// DMA_MM2S_Status_0 at 0x1D228 (119336) for tile (0,0) channel 0.

// Only the fifth push can overflow, so exactly one poll is emitted, against the
// status register and depth bit for this channel.
// CHECK-LABEL: @enforce
// CHECK-DAG:   arith.constant 119336 : i32
// CHECK-DAG:   arith.constant 4194304 : i32
// CHECK:       aiex.npu.maskpoll
// CHECK-NOT:   aiex.npu.maskpoll

// Off by default: codegen is untouched.
// OFF-LABEL: @enforce
// OFF-NOT:   aiex.npu.maskpoll
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @enforce(%arg0: memref<1280xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 512 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
    %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 768 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t3)
    %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1280xi32> offset = 1024 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t4)
  }
}
