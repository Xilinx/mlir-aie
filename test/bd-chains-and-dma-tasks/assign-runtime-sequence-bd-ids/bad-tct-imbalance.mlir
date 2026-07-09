//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --verify-diagnostics --split-input-file %s

// A task-completion-token (TCT) await with no matching issue_token push on its
// channel deadlocks the host: the sync runs on an empty FIFO and blocks the
// runtime sequence forever. After runtime control flow is rejected and constant
// loops are unrolled, the check is a straight-line per-channel token count, so
// these are caught as under-production.

// One push, four awaits: the loop over-consumes. Unrolled, the channel sees a
// single issue_token start and four syncs; the second sync has no token.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @loop_await_extern(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error@+1 {{would block here forever}}
      aiex.dma_await_task(%t)
    }
  }
}



// -----

// A produce in an inner loop, awaited more times than produced in the outer
// body: after unrolling every produce becomes straight-line, and the running
// per-channel count still goes negative at the offending await.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @nested_loop_extern(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      scf.for %j = %c0 to %c4 step %c1 {
        // expected-error@+1 {{would block here forever}}
        aiex.dma_await_task(%t)
      }
    }
  }
}



// -----

// A configure that is awaited but never started: the FIFO is empty at the sync.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @await_without_start(%arg0: memref<8xi16>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
      aie.end
    } {issue_token = true}
    // expected-error@+1 {{would block here forever}}
    aiex.dma_await_task(%t)
  }
}
