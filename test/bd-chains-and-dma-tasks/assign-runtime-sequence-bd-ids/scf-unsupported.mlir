//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// Control-flow forms whose BD-ID write-back needs a runtime-selected BD id are
// rejected with a clear diagnostic rather than miscompiled (or crashing, as the
// program-order allocator did). These become supported once a runtime bd_id
// representation lands.

// Rolled ping-pong: handle freed in a later iteration (crosses the back-edge).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %c4 step %c1 iter_args(%prev = %init) -> (index) {
      // expected-error@+1 {{held across a loop back-edge}}
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

// Task configured in a loop with no reachable completion sync (leaks one BD per
// iteration).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error@+1 {{configured in a loop is never completed}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Task handle yielded out of an scf.if arm (escapes via a value join).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>, %c: i1) {
    // expected-error@+1 {{escapes its control-flow region via an scf.if result}}
    %x = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    %r = scf.if %c -> (index) {
      scf.yield %x : index
    } else {
      %z = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%z)
      scf.yield %z : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Handle carried UNCHANGED across a loop back-edge (yield the iter_arg itself).
// This is a def-use cycle; the analysis must terminate and reject, not hang.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{cannot be statically resolved to a single completion}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %init) -> (index) {
      scf.yield %p : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Handle both freed and re-yielded across the back-edge in the same iteration:
// two continuations, cannot be linearized => reject (must not crash).
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{cannot be statically resolved to a single completion}}
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%init)
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %init) -> (index) {
      aiex.dma_free_task(%p)
      scf.yield %p : index
    }
    aiex.dma_free_task(%r)
  }
}

// -----

// Free of a value that is not a dma_configure_task result (here, an scf.for
// result). Must be rejected cleanly, not crash the recycle path.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %c0) -> (index) {
      scf.yield %p : index
    }
    // expected-error@+1 {{does not reference a valid configure_task operation}}
    aiex.dma_free_task(%r)
  }
}
