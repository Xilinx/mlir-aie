//===- cross_mem_op_bd_id_collision.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// A user-assigned bd_id that collides with one already taken on the same tile
// must be rejected even when the two BDs live in different aie.mem regions.
// The allocator tracks ids per tile, so it can see across regions; a per-region
// allocator would miss this and let the second BD overwrite the first's slot.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>

    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }

    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd1, ^end)
      ^bd1:
        // expected-error@+1 {{assigned bd_id 0 is already used by another BD on this tile}}
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// Different tiles have independent BD tables: the same id on each is fine.
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %b0 = aie.buffer(%t0) : memref<8xi32>
    %b1 = aie.buffer(%t1) : memref<8xi32>

    aie.mem(%t0) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%b0 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }

    aie.mem(%t1) {
        aie.dma_start(S2MM, 0, ^bd1, ^end)
      ^bd1:
        aie.dma_bd(%b1 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
