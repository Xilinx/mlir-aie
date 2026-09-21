//===- dma_out_of_order_channel_boundary.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// Only the out-of-order BD (bd0) is unchained; the in-order channel's BDs
// (bd1, bd2) still chain (enableNextBd=1).
// CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd0){{.*}}/* enableNextBd */ 0));
// CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd1){{.*}}/* enableNextBd */ 1));
// CHECK: XAie_DmaSetNextBd(&(dma_tile02_bd2){{.*}}/* enableNextBd */ 1));

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "buf" } : memref<8xi32>
    %m = aie.mem(%t) {
      %s0 = aie.dma_start(S2MM, 0, ^ooo0, ^dma1, repeat_count = 1) { out_of_order }
    ^dma1:
      %s1 = aie.dma_start(MM2S, 1, ^io0, ^end)
    ^ooo0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
      aie.next_bd ^io0
    ^io0:
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 1 : i32 }
      aie.next_bd ^io1
    ^io1:
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%b : memref<8xi32> offset = 4 len = 4) { bd_id = 2 : i32 }
      aie.next_bd ^io0
    ^end:
      aie.end
    }
  }
}
