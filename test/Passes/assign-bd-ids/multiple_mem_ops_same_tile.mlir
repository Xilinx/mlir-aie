//===- multiple_mem_ops_same_tile.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids %s | FileCheck %s

// A tile may be described by more than one aie.mem region -- e.g. a design
// that builds two TileDma programs for the same tile, or hand-written IR that
// keeps an input channel and an output channel in separate regions. BD ids are
// a per-*tile* resource (one BD table shared by every channel on the tile), so
// the allocator must not restart numbering for each region: two BDs sharing an
// id silently overwrite each other's slot in the BD table, which corrupts
// whichever channel loses the race with no diagnostic at all.

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>

    // CHECK: aie.mem
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }

    // Must NOT restart at 0: this BD lives in the same tile's BD table.
    // CHECK: aie.mem
    // CHECK-NOT: aie.dma_bd({{.*}}) {bd_id = 0 : i32
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32
    aie.mem(%t) {
        aie.dma_start(MM2S, 0, ^bd1, ^end)
      ^bd1:
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
