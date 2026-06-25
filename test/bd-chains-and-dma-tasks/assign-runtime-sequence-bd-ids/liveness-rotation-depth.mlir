//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-test-runtime-bd-liveness --verify-diagnostics --split-input-file %s

// Unit test for the control-flow-aware runtime-sequence BD liveness analysis
// (resolveTaskLiveRange). Verifies, per dma_configure_task, the completion-sync
// kind, the number of scf.for back-edges the live handle crosses, and whether
// the handle leaks (no sync reachable). Independent of BD-ID coloring and the
// dynamic bd_id write-back form.

//===----------------------------------------------------------------------===//
// Straight-line forms
//===----------------------------------------------------------------------===//

// Explicit free => backedges=0, kill is dma_free_task.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    aiex.dma_free_task(%t1)
  }
}

// -----

// Await => backedges=0, kill is dma_await_task.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_await_task}}
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
    aiex.dma_await_task(%t1)
  }
}

// -----

// No sync at top level => leaked (legal: BD simply stays occupied, as in
// designs that allocate all BDs without freeing). Not in a loop, so no error.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    // expected-remark@+1 {{bd-liveness: leaked}}
    %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%t1)
  }
}

//===----------------------------------------------------------------------===//
// Loop forms
//===----------------------------------------------------------------------===//

// -----

// scf.for, freed within the same iteration => backedges=0, in-loop.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_await_task in-loop}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Ping-pong: free the previous iteration's task via iter_args => backedges=1.
// Prologue task is carried in as the iter_arg init (depth-2 working set: the
// prologue and the first in-body task coexist).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // Prologue is freed as %prev in iteration 0 (before any back-edge).
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      // expected-remark@+1 {{bd-liveness: backedges=1 kill=aiex.dma_free_task in-loop}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_free_task(%last)
  }
}

// -----

// Free-two-iterations-ago: two iter_args carry the two most-recent tasks, the
// oldest freed each iteration => working set 3 (backedges 0/1 for the two
// prologue tasks, 2 for the in-body task).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task}}
    %i0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    // expected-remark@+1 {{bd-liveness: backedges=1 kill=aiex.dma_free_task}}
    %i1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%p2 = %i0, %p1 = %i1) -> (index, index) {
      // expected-remark@+1 {{bd-liveness: backedges=2 kill=aiex.dma_free_task in-loop}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%p2)
      scf.yield %p1, %t : index, index
    }
    aiex.dma_free_task(%r#0)
    aiex.dma_free_task(%r#1)
  }
}

// -----

// Nested loops, inner ping-pong fully closed within each outer iteration =>
// inner tasks backedges 0/1 (the outer loop does not extend their ranges).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %o = %c0 to %c4 step %c1 {
      // Inner prologue freed as %prev in inner iteration 0 (no inner back-edge).
      // expected-remark@+1 {{bd-liveness: backedges=0 kill=aiex.dma_free_task in-loop}}
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
        // expected-remark@+1 {{bd-liveness: backedges=1 kill=aiex.dma_free_task in-loop}}
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
          aie.end
        }
        aiex.dma_start_task(%t)
        aiex.dma_free_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_free_task(%last)
    }
  }
}
