//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics --split-input-file %s

// aiex.dma_free_task returns BD ids to the allocator with no completion
// guarantee, and nextBdId scans upward from 0, so a just-freed low id is the
// first one handed out again -- the worst case for reprogramming a BD that is
// still running. The warning fires at the reuse, not the free: releasing an id
// is only a problem once something else takes it.

// Freed with no completion guarantee at all, then reused.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @unsafe(%arg0: memref<512xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
      aie.dma_bd(%arg0 : memref<512xi32> offset = 256 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t1)
  }
}

// -----

// Awaiting a LATER task on the same channel covers everything queued ahead of
// it, because a channel completes its tasks in order. This is the idiom
// DMATasks.md blesses, and it must stay quiet.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @safe_by_channel_order(%arg0: memref<512xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 256 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t1)
    aiex.dma_await_task(%t1)
    aiex.dma_free_task(%t0)
    %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t2)
  }
}
