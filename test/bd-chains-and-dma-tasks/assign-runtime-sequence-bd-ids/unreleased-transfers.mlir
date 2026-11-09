// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --aie-assign-runtime-sequence-bd-ids --verify-diagnostics %s -o /dev/null
// RUN: aie-opt --split-input-file --aie-assign-runtime-sequence-bd-ids="warn-unreleased-transfers=false" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty
// QUIET-NOT: warning:

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @leaked_token(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    // expected-note@+1 {{first outstanding transfer starts here}}
    aiex.dma_start_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @leaked_no_token(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    // expected-note@+1 {{first outstanding transfer starts here}}
    aiex.dma_start_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @awaited(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @freed(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    aiex.dma_start_task(%a)
    aiex.dma_free_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @configured_only(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @await_consumes_oldest(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    %b = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    // expected-note@+1 {{first outstanding transfer starts here}}
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @repeated_start(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    // expected-note@+1 {{first outstanding transfer starts here}}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @await_covers_non_token_prefix(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    %b = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed on tile (0,0) MM2S channel 0}}
  aie.runtime_sequence @raw_leak(%input : memref<16xi32>) {
    // expected-note@+1 {{first outstanding transfer starts here}}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64} : memref<16xi32>
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @raw_awaited(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @alloc}
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @raw_prefix(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64} : memref<16xi32>
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @polled(%input : memref<16xi32>) {
    %a = aiex.dma_configure_task(%shim, MM2S, 0) {
      aie.dma_bd(%input : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    aiex.dma_start_task(%a)
    %status = arith.constant 119336 : i32
    %zero = arith.constant 0 : i32
    %idle = arith.constant 7864380 : i32
    aiex.npu.maskpoll(%status, %zero, %idle) : i32, i32, i32
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc(%shim, MM2S, 0)
  aie.runtime_sequence @raw_synced(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @alloc, id = 0 : i64, issue_token = true} : memref<16xi32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

aie.device(npu2) {
  %shim = aie.tile(0, 0)
  // expected-warning@+1 {{may exit with 1 DMA transfer(s) neither awaited nor freed}}
  aie.runtime_sequence @raw_push() {
    %c0 = arith.constant 0 : i32
    // expected-note@+1 {{first outstanding transfer starts here}}
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
  aie.runtime_sequence @empty_after_leak() {
  }
}
