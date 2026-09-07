//===- dma_out_of_order_pinned_bd_id.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// Check bd id pinning (not positional).
// CHECK-NOT: dma_tile02_bd0
// CHECK-NOT: dma_tile02_bd1
// CHECK: XAie_DmaDesc dma_tile02_bd3;
// CHECK: XAie_DmaWriteBd({{.*}}&(dma_tile02_bd3){{.*}}/* bd */ 3));
// CHECK: XAie_DmaDesc dma_tile02_bd7;
// CHECK: XAie_DmaWriteBd({{.*}}&(dma_tile02_bd7){{.*}}/* bd */ 7));

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "buf" } : memref<8xi32>
    %m = aie.mem(%t) {
      %s0 = aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 1) { out_of_order }
    ^bd0:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 3 : i32 }
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%b : memref<8xi32> offset = 4 len = 4) { bd_id = 7 : i32 }
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}
