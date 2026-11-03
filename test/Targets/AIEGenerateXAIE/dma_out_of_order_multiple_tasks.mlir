// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-bd-ids --aie-prepare-buffers --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// Repeated submissions to one channel must use separate C variable scopes.
// CHECK:      XAie_DmaWriteBd
// CHECK-NEXT: {
// CHECK-NEXT: XAie_DmaChannelDesc ooo_desc_0_2_0;
// CHECK:      XAie_DmaDeclareQueueConfig(ooo_queue_0_2_0, {{.*}}/* Repeat */ 256,
// CHECK:      }
// CHECK:      {
// CHECK-NEXT: XAie_DmaChannelDesc ooo_desc_0_2_0;
// CHECK:      XAie_DmaDeclareQueueConfig(ooo_queue_0_2_0, {{.*}}/* Repeat */ 44,
// CHECK:      }
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
      aie.dma_start(S2MM, 0, ^bd, ^next, repeat_count = 255) {out_of_order}
    ^next:
      aie.dma_start(S2MM, 0, ^bd, ^end, repeat_count = 43) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b : memref<8xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
