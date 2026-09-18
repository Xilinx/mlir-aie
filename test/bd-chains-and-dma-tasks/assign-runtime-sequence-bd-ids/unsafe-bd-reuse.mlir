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

// -----

// Two sequences in one device. Each is a separate dispatch, and BD id
// allocation restarts for each, so an id freed in flight in @first says
// nothing about the same id in @second -- warning there would be a false
// positive. It is also the free in @first that the note would point at, and
// that op is erased once @first is done, so the stale entry dangles.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @first(%arg0: memref<512xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
    aiex.dma_free_task(%t0)
  }
  aie.runtime_sequence @second(%arg0: memref<512xi32>) {
    %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<512xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%t0)
  }
}

// -----

// A wait after the free, but before reuse, proves the released task complete.
// Reserve a different ID for the token task so configuring it is not the reuse.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @free_then_await(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a)
    aiex.dma_free_task(%a)
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// A configure that was never started cannot have descriptors in flight.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @never_started(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
    aiex.dma_free_task(%a)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Both outstanding starts must complete before a released ID is safe to reuse.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @free_then_sync_twice(%buf: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%a)
    aiex.dma_free_task(%a)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Consuming only the first token leaves the second start in flight.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @free_then_sync_once(%buf: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%a)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%a)
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0)}}
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// A token wait on another channel cannot prove the released task complete.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @free_then_await_other_channel(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%a)
    %b = aiex.dma_configure_task(%tile, MM2S, 1) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0)}}
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// A memcpy token and wait also establish completion of a freed earlier task.
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @free_then_memcpy_wait(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a)
    aiex.dma_free_task(%a)
    aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 7 : i64, metadata = @alloc, issue_token = true} : memref<256xi32>
    aiex.npu.dma_wait {symbol = @alloc}
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}
