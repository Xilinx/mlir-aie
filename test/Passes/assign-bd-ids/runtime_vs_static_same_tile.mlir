//===- runtime_vs_static_same_tile.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A tile's BD table is shared between the statically-configured DMAs numbered
// by --aie-assign-bd-ids and the runtime-sequence tasks numbered by
// --aie-assign-runtime-sequence-bd-ids. The runtime allocator therefore has to
// start from the ids the static one already took on that tile; if both start
// at 0, the runtime task's BD silently overwrites the static BD's slot and
// whichever channel loses the race is corrupted with no diagnostic.

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %sb = aie.external_buffer {sym_name = "sb"} : memref<8xi32>

    // Static shim BD takes id 0.
    // CHECK: aie.shim_dma
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
    aie.shim_dma(%shim) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%sb : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }

    // Runtime task on the SAME tile must not reuse id 0.
    // CHECK: aie.runtime_sequence
    // CHECK-NOT: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %t = aiex.dma_configure_task(%shim, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 4)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
