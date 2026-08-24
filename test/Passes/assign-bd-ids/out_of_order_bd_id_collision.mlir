//===- out_of_order_bd_id_collision.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// Two out-of-order S2MM channels on the same tile must use disjoint ids.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^dma1, repeat_count = 1) { out_of_order }
      ^dma1:
        aie.dma_start(S2MM, 1, ^bd1, ^end, repeat_count = 1) { out_of_order }
      ^bd0:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^bd1:
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{assigned bd_id 0 is already used by another BD on this tile}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
