//===- dma_out_of_order_two_tile.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two out-of-order S2MM channels on different tiles that share a channel index
// (both channel 0) are emitted into one libxaie function body. The per-channel
// start-queue locals must be named uniquely per tile+channel.

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// CHECK-DAG: XAie_DmaChannelDesc ooo_desc_0_2_0;
// CHECK-DAG: XAie_DmaChannelDesc ooo_desc_1_2_0;
// CHECK-DAG: XAie_DmaDeclareQueueConfig(ooo_queue_0_2_0,
// CHECK-DAG: XAie_DmaDeclareQueueConfig(ooo_queue_1_2_0,

module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 2)
    %b0 = aie.buffer(%t0) : memref<8xi32>
    aie.mem(%t0) {
        aie.dma_start(S2MM, 0, ^bd0, ^end0, repeat_count = 7) { out_of_order }
      ^bd0:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b0 : memref<8xi32> offset = 4 len = 4) { bd_id = 1 : i32 }
        aie.next_bd ^end0
      ^end0:
        aie.end
    }
    %t1 = aie.tile(1, 2)
    %b1 = aie.buffer(%t1) : memref<8xi32>
    aie.mem(%t1) {
        aie.dma_start(S2MM, 0, ^bd2, ^end1, repeat_count = 9) { out_of_order }
      ^bd2:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b1 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b1 : memref<8xi32> offset = 4 len = 4) { bd_id = 1 : i32 }
        aie.next_bd ^end1
      ^end1:
        aie.end
    }
  }
}
