//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics --split-input-file %s | FileCheck %s

// Naming the newest task does not skip the oldest token. The middle task is
// still in flight, and the named task's BD must not be implicitly recycled.
// CHECK-LABEL: @oldest_token_only
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aie.dma_bd({{.*}} {bd_id = 2 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aie.dma_bd({{.*}} {bd_id = 3 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @oldest_token_only(%buf: memref<256xi32>) {
    %oldest = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %middle = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %newest = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%oldest)
    aiex.dma_start_task(%middle)
    aiex.dma_start_task(%newest)
    aiex.dma_await_task(%newest)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%middle)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 1 on tile (0,0)}}
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %still_live = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }

  // -----

  // A raw channel sync consumes the same FIFO token as a task await and must
  // release a pending task once that token proves it has completed.
  // CHECK-LABEL: @sync_completes_pending
  // CHECK: aiex.dma_await_task
  // CHECK: aiex.npu.sync
  // CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.runtime_sequence @sync_completes_pending(%buf: memref<256xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %a = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      %b = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%a)
      aiex.dma_start_task(%b)
      aiex.dma_await_task(%b)
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
        aie.end
      }
    }
  }
}

// -----

// Conversely, naming the oldest configure twice consumes both tokens. The
// second await proves completion of the non-token task between them.
// CHECK-LABEL: @two_tokens
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @two_tokens(%buf: memref<256xi32>) {
    %oldest = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    %middle = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    }
    %newest = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 2 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%oldest)
    aiex.dma_start_task(%middle)
    aiex.dma_start_task(%newest)
    aiex.dma_await_task(%oldest)
    aiex.dma_await_task(%oldest)
    aiex.dma_free_task(%middle)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    }
  }
}

// -----

// Completion of one start does not complete another outstanding start of the
// same configure. Its descriptor still belongs to the second queued transfer.
// CHECK-LABEL: @repeated_start
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @repeated_start(%buf: memref<256xi32>) {
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_await_task(%task)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%task)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 0 on tile (0,0)}}
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }

}

// -----

// The second await consumes B's token, but names A whose ID now belongs to C.
// Neither that await nor a redundant explicit free may release C's ID.
// CHECK-LABEL: @redundant_releases_after_reuse
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 2 : i32}
// CHECK: aie.dma_bd({{.*}} {bd_id = 3 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @redundant_releases_after_reuse(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%a)
    %c = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_await_task(%a)
    %d = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_free_task(%a)
    %e = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// Restarting a configure after an await keeps its ID reserved until the final
// start completes, even when another configure is allocated in between.
// CHECK-LABEL: @restart_after_await
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @restart_after_await(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
    %c = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}

// -----

// Both configures have been awaited and completed after consuming both tokens,
// even though each await consumed the other configure's token.
// CHECK-LABEL: @permuted_awaits
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @permuted_awaits(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
    aiex.dma_await_task(%a)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    }
  }
}

// -----

// Explicitly freeing a pending configure cancels its deferred release. When
// its token is later consumed, the recycled ID must retain its new owner.
// CHECK-LABEL: @free_pending_await
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
// CHECK: aie.dma_bd({{.*}} {bd_id = 2 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @free_pending_await(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
    // expected-note@+1 {{released here}}
    aiex.dma_free_task(%b)
    %c = aiex.dma_configure_task(%tile, MM2S, 0) {
      // expected-warning@+1 {{reuses buffer descriptor ID 1 on tile (0,0)}}
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    }
    aiex.dma_await_task(%a)
    %d = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %e = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
  }
}
